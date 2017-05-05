#include "StdAfx.h"
#include "GenericParticleFilter.h"

GenericParticleFilter::GenericParticleFilter(IplImage* frame, int numParticles, double sigma) { 
	// khoi tao 
	particles = new PointParticle[numParticles];
	previousFrame = cvCreateImage(cvGetSize(frame), frame->depth, frame->nChannels);	

	// nap thong so tu input vao GenericParticleFilter	
	this->numParticles = numParticles;
	this->sigma = sigma;
	cvCopyImage(frame, previousFrame);

	initParticles();
}
GenericParticleFilter::~GenericParticleFilter(void) { 
	cvReleaseImage(&previousFrame);
	delete[] particles;
}

// ============================================================================================

IplImage* GenericParticleFilter::process(IplImage* currFrame) {

	// --------------------------------------------------------------
	// measurementFrame dung de do su chenh lech giua 2 frame lien tiep
	IplImage *measurementFrame = cvCreateImage(cvGetSize(currFrame), currFrame->depth, currFrame->nChannels);
	measurementFrame->origin = currFrame->origin; // currFrame->origin = 1 ==> bottom-left origin (Windows bitmaps style) 
	cvAbsDiff(currFrame, previousFrame, measurementFrame); // absoluted subtraction
	
	// --------------------------------------------------------------
	// da initParticles o constructor
	addNoiseToParticles();
	getIntensityForParticles(measurementFrame);
	resampling();

	cvCopyImage(currFrame, previousFrame);

	// --------------------------------------------------------------
	// ve particles len currFrame
	for (int i = 0; i < numParticles; i++) 
		cvCircle(currFrame, cvPoint(particles[i].x, particles[i].y), 1, CV_RGB(255,0,0));

	// --------------------------------------------------------------
	cvReleaseImage(&measurementFrame);
	return currFrame;
}

void GenericParticleFilter::initParticles() { 

	// phan bo deu cac particles tren anh
	int imgWidth = previousFrame->width, imgHeight = previousFrame->height;
	int areaOfImg = imgWidth * imgHeight;
	int areaForAParticle = int(double(areaOfImg) / numParticles);
	int edgeOfAreaForAParticle = (int)sqrt(double(areaForAParticle)); // xem nhu moi area la hinh vuong
	int numParticlesOnARow = int(double(imgWidth) / edgeOfAreaForAParticle);
	int numParticlesOnAColumn = int(double(imgHeight) / edgeOfAreaForAParticle);

	while (numParticlesOnARow*numParticlesOnAColumn < numParticles) { // vi lam tro`n xuo^'ng, nen phai bu` lai cho ddu? particles
		edgeOfAreaForAParticle--;
		numParticlesOnARow = int(double(imgWidth) / edgeOfAreaForAParticle);
		numParticlesOnAColumn = int(double(imgHeight) / edgeOfAreaForAParticle);
	}

	// --------------------------------------------------------------
	// init particles' positions, weights
	int countParticles = 0;
	for (int i = 0; i < numParticlesOnARow; i++) 
		for (int j = 0; j < numParticlesOnAColumn; j++) 
			if (countParticles < numParticles) {
				particles[countParticles].x = j*edgeOfAreaForAParticle + edgeOfAreaForAParticle/2;
				particles[countParticles].y = i*edgeOfAreaForAParticle + edgeOfAreaForAParticle/2;
				particles[countParticles].weight = 1.0/numParticles;
				countParticles++;
			}

	// --------------------------------------------------------------
	// init particles's Gaussian Random Number Generator, bo tao so ngau nhien nay dung de ta.o nhieu cho position cua particles
	double seed;
	srand((unsigned)time(NULL)); // dung de phat sinh so ngau nhien can cu theo clock cua CPU
	for(int i = 0; i < numParticles; i++){
		// moi particle se co 1 bo^. phat sinh so ngau nhien rieng biet & ko thay doi bo^. phat sinh nay theo thoi gian
		seed = rand()*1.0/RAND_MAX;		particles[i].gaussianGenX.setSeed(seed, 0, sigma);
		seed = rand()*1.0/RAND_MAX;		particles[i].gaussianGenY.setSeed(seed, 0, sigma);
	}
}

void GenericParticleFilter::addNoiseToParticles() {
	int dx, dy; // dx, dy o day la bien nhie^~u v nhu trong sach, no duoc dung de co^.ng them vao x, y
	for(int i = 0; i < numParticles; i++){
		dx = (int)(particles[i].gaussianGenX.rnd() + 0.5); // dung bo^. phat sinh so ngau nhieu o moi particle de phat sinh so ngau nhien
		dy = (int)(particles[i].gaussianGenY.rnd() + 0.5);

		// nhieu duoc cong them vao position 
		particles[i].x += dx; 
		particles[i].y += dy;
	}
}

void GenericParticleFilter::getIntensityForParticles(IplImage* observedFrame) {
	int height = observedFrame->height, width = observedFrame->width, step = observedFrame->widthStep, channels = observedFrame->nChannels;
	const uchar* observedFrameData = (uchar *)observedFrame->imageData;

	for (int i = 0; i < numParticles; i++) {
		int particleX = particles[i].x;
		int particleY = particles[i].y;
		if (particleX < 0 || width <= particleX || particleY < 0 || height <= particleY)
			particles[i].intensity = 0;
		else {
			if (channels == 1)
				particles[i].intensity = observedFrameData[particleY*step + particleX];
			else { // channels == 3 
				int b = observedFrameData[particleY*step + particleX*channels];
				int g = observedFrameData[particleY*step + particleX*channels + 1];
				int r = observedFrameData[particleY*step + particleX*channels + 2];
				particles[i].intensity = int((b + g + r)*1.0/3);
			}
		}
	}
}

void GenericParticleFilter::resampling() {

	// --------------------------------------------------------------
	// tinh tong tat ca intensity cua particles
	// intensity o day that chat la weight chua chuan hoa
	int cummulativeSumOfWeights = 0;
	for (int i = 0; i < numParticles; i++) 
		cummulativeSumOfWeights += particles[i].intensity;
	if (cummulativeSumOfWeights == 0) return;

	// --------------------------------------------------------------
	// xac dinh ddo^. tra?i cua moi particle thong qua intensity cua no
	// vd: particle1 co intensity la 5 thi particlesSpreadViaItsIntensity[] = {1, 1, 1, 1, 1} // 5 so 1 tuong ung voi particle1
	//     tiep theo, particle2 co intensity là 3 thi particlesSpreadViaItsIntensity[] = {1, 1, 1, 1, 1, 2, 2, 2}
	// ... nhu vay la neu nhu particle nao co intensity cang lon thi spread cua no cang ro^.ng, lam cho xac suat de chon no lo*'n
	int* particlesSpreadViaItsIntensity = new int[cummulativeSumOfWeights];
	int index = 0;
	for (int i = 0; i < numParticles; i++) {
		int intensityOfAParticle = particles[i].intensity;
		for (int j = 0; j < intensityOfAParticle; j++) 
			particlesSpreadViaItsIntensity[index++] = i;
	}

	// --------------------------------------------------------------
	PointParticle* tmpParticles = new PointParticle[numParticles];
	srand((unsigned)time(NULL)); // su dung truoc khi goi ham rand()

	for (int i = 0; i < numParticles; i++) {
		index = particlesSpreadViaItsIntensity[ int( (rand()*1.0/RAND_MAX) * (cummulativeSumOfWeights - 1) + 0.5 ) ]; // index la ngau nhien
		
		tmpParticles[i].x = particles[index].x;
		tmpParticles[i].y = particles[index].y;

		// that ra thi cung ko can ga'n intensity lai lam gi, vi trong process() se lay lai intensity cho toan bo cac particles mo*'i cap nhat
//		tmpParticles[i].intensity = particles[index].intensity; 

		// weight o day duoc tinh dua tren intensity thoi, & ko co cung duoc // chang qua la muon minh hoa cho normalized weights
//		tmpParticles[i].weight = particles[index].intensity * 1.0 / cummulativeSumOfWeights; 

		// ga'n lai bo phat sinh so ngau nhien cu
		tmpParticles[i].gaussianGenX = particles[i].gaussianGenX;
		tmpParticles[i].gaussianGenY = particles[i].gaussianGenY;
	}

	// --------------------------------------------------------------
	delete[] particlesSpreadViaItsIntensity;
	delete[] particles;
	particles = tmpParticles;
}