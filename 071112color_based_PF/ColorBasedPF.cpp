#include "StdAfx.h"
#include "ColorBasedPF.h"

ColorBasedPF::ColorBasedPF(void /* param giong setParamsAndInit */) { }

ColorBasedPF::~ColorBasedPF(void) { 
	delete[] objHistogram;
	delete[] particles;
}

void ColorBasedPF::setParamsAndInit(int numParticles, int hx, int hy, CvPoint expectedPositionOfObj, double positionSigma, double velocitySigma,
									double sizeSigma, IplImage* initFrame) { // giong constructor
	// khoi tao 
	particles = new EllipseParticle[numParticles];

	// nap thong so tu input vao ColorBasedPF
	this->numParticles = numParticles;	
	this->hx = hx;
	this->hy = hy;
	this->expectedPositionOfObj = cvPoint(expectedPositionOfObj.x, expectedPositionOfObj.y);

	this->positionSigma = positionSigma;
	this->velocitySigma = velocitySigma;
	this->sizeSigma = sizeSigma;
	
	// init
	objHistogram = calculateRGBHistogram(initFrame, expectedPositionOfObj, hx, hy);
	initParticles();
}

// ============================================================================================

void ColorBasedPF::process(IplImage* currFrame) {
	predict(currFrame->width, currFrame->height);
	calculateParticlesInfo(currFrame);	// = observation = measurement

	//writeInfoToFile();	
	//drawAllParticles(currFrame);

	drawMeanState(currFrame, estimateMeanState());
	resamplingCBPF();
}

void ColorBasedPF::initParticles() {
	/* khoi tao x,y,hX,hY cho cac particles ==> phat sinh cac particles (sap xep ngau nhien) 
	   tai vung du doan la obj xuat hien (hay chuyen dong) (xem Fig. 4)	*/

	// init particles's Gaussian Random Number Generator, bo tao so ngau nhien nay dung de ta.o nhieu cho position cua particles
	double seed;
	srand((unsigned)time(NULL)); // dung de phat sinh so ngau nhien can cu theo clock cua CPU
	for(int i = 0; i < numParticles; i++){
		// moi particle se co 1 bo^. phat sinh so ngau nhien rieng biet & ko thay doi bo^. phat sinh nay theo thoi gian
		seed = rand()*1.0/RAND_MAX;		particles[i].gaussianGenX.setSeed(seed, 0, positionSigma);
		seed = rand()*1.0/RAND_MAX;		particles[i].gaussianGenY.setSeed(seed, 0, positionSigma);
		seed = rand()*1.0/RAND_MAX;		particles[i].gaussianGenVelocityX.setSeed(seed, 0, velocitySigma);
		seed = rand()*1.0/RAND_MAX;		particles[i].gaussianGenVelocityY.setSeed(seed, 0, velocitySigma);
		seed = rand()*1.0/RAND_MAX;		particles[i].gaussianGenHx.setSeed(seed, 0, sizeSigma);
		seed = rand()*1.0/RAND_MAX;		particles[i].gaussianGenHy.setSeed(seed, 0, sizeSigma);
	}

	// init particles' positions
	for(int i = 0; i < numParticles; i++){
		// chia cho 4 la de tranh truong hop ddu.ng bie^n
		particles[i].centerX = expectedPositionOfObj.x + int( (particles[i].gaussianGenX.rnd() + 0.5) / 4 );
		particles[i].centerY = expectedPositionOfObj.y + int( (particles[i].gaussianGenY.rnd() + 0.5) / 4 );

		//
		particles[i].velocityX = int(particles[i].gaussianGenVelocityX.rnd());
		particles[i].velocityY = int(particles[i].gaussianGenVelocityY.rnd());

		// tranh truong hop hx, hy <= 0 va hx, hy > centerX, centerY (neu ko thi ko tinh histogram duoc)
		particles[i].hx = abs(hx + int(particles[i].gaussianGenHx.rnd() + 0.5)) + 1;
		particles[i].hy = abs(hy + int(particles[i].gaussianGenHy.rnd() + 0.5)) + 1;
		if (particles[i].hx > particles[i].centerX) particles[i].hx = particles[i].centerX;
		if (particles[i].hy > particles[i].centerY) particles[i].hy = particles[i].centerY;
	}
}

void ColorBasedPF::predict(int imgWidth, int imgHeight) { 
	/*   = propagate (using a linear stochastic differential equation, Eq. 9)
		"the noise terms are chosen proportional to the size of the initial region"  */
	
	int dx, dy, dhx, dhy; // dx, dy o day la bien nhie^~u v nhu trong sach, no duoc dung de co^.ng them vao x, y
	for(int i = 0; i < numParticles; i++){
		//particles[i].centerX += int(particles[i].gaussianGenX.rnd()); // old dynamic model 
		//particles[i].centerY += int(particles[i].gaussianGenY.rnd());

		// --- position ---
		dx = particles[i].velocityX + int( (particles[i].gaussianGenX.rnd() + 0.5) / 2 );
		dy = particles[i].velocityY + int( (particles[i].gaussianGenY.rnd() + 0.5) / 2 );
		
		particles[i].centerX += dx; 
		particles[i].centerY += dy;
		
		if (particles[i].centerX < hx || imgWidth - hx < particles[i].centerX) particles[i].centerX -= dx;
		if (particles[i].centerY < hy || imgHeight - hy < particles[i].centerY) particles[i].centerY -= dy;

		// --- velocity ---
		particles[i].velocityX += int(particles[i].gaussianGenVelocityX.rnd() + 0.5);
		particles[i].velocityY += int(particles[i].gaussianGenVelocityY.rnd() + 0.5);

		// --- size ---
		dhx = int(particles[i].gaussianGenHx.rnd());
		dhy = int(particles[i].gaussianGenHy.rnd());
		particles[i].hx += dhx;
		particles[i].hy += dhy;
		if (particles[i].hx <= 0 || particles[i].hx > particles[i].centerX) particles[i].hx -= dhx; // tranh truong hop hx, hy <= 0 (neu ko thi ko tinh histogram duoc)
		if (particles[i].hy <= 0 || particles[i].hy > particles[i].centerY) particles[i].hy -= dhy;
	}
}	

void ColorBasedPF::calculateParticlesInfo(IplImage* currFrame) { 
	/*	 = observation = measurement
		moi khi thay doi vi tri cua particles thi phai tinh lai colorDistribution, BhaCoeff, weight */

	for(int i = 0; i < numParticles; i++){
		double* colorDistribution = calculateRGBHistogram(currFrame, cvPoint(particles[i].centerX, particles[i].centerY), hx, hy);
		particles[i].weight = weight(bhattacharyyaCoefficient(colorDistribution, objHistogram), 0.5); // chon sigma = 0.1 la kha' tot
		delete[] colorDistribution;

		/*if ( (particles[i].centerX - particles[i].hx >= 0) && (particles[i].centerY - particles[i].hy >= 0) 
			  && (particles[i].centerX + particles[i].hx < currFrame->width) && (particles[i].centerY + particles[i].hy < currFrame->height) ) {

			double* colorDistribution = calculateRGBHistogram(currFrame, cvPoint(particles[i].centerX, particles[i].centerY), particles[i].hx, particles[i].hy);
			particles[i].weight = weight(bhattacharyyaCoefficient(colorDistribution, objHistogram), 0.5); // chon sigma = 0.1 la kha' tot
			delete[] colorDistribution;

		} else {
			particles[i].weight = 0;
			if (particles[i].centerX < 0) particles[i].centerX = 0;
			if (particles[i].centerY < 0) particles[i].centerY = 0;
			if (particles[i].hx < 0) particles[i].hx = 0;
			if (particles[i].hy < 0) particles[i].hy = 0;
		}*/
	}	
	normalizeWeights(particles);
}

CvRect ColorBasedPF::estimateMeanState() {
	double meanX = 0, meanY = 0, meanHx = 0, meanHy = 0;
	for (int i = 0; i < numParticles; i++) {
		meanX += (particles[i].centerX * particles[i].weight);
		meanY += (particles[i].centerY * particles[i].weight);
		meanHx += (particles[i].hx * particles[i].weight);
		meanHy += (particles[i].hy * particles[i].weight);
	}
	return cvRect(int(meanX), int(meanY), int(meanHx), int(meanHy));
}

void ColorBasedPF::resamplingCBPF() { // resampling nhu trong paper Color-based PF
	
	// ---- khai bao bien ----
	EllipseParticle* tmpParticles = new EllipseParticle[numParticles];

	srand((unsigned)time(NULL)); // su dung truoc khi goi ham rand()
	CvRNG rng_state = cvRNG(rand());

	//CStdioFile f;	f.Open(L"resamplingIndex.txt", CFile::modeCreate | CFile::modeWrite);
	//CString text;

	// ---- init cummulativeSumOfWeights ----
	double* cummulativeSumOfWeights = new double[numParticles];
	cummulativeSumOfWeights[0] = particles[0].weight;
	for (int i = 1; i < numParticles; i++) 
		cummulativeSumOfWeights[i] = cummulativeSumOfWeights[i-1] + particles[i].weight;

	// ---- main resampling ----
	int index = -1;
	for (int i = 0; i < numParticles; i++) {
		double r =  cvRandReal(&rng_state);

		for (int j = 0; j < numParticles; j++) {// search the smallest j for which c[j] > r
			//text.Format(L"%3.3f \n", cummulativeSumOfWeights[j]);	f.WriteString(text);
			if (cummulativeSumOfWeights[j] > r) {
				index = j;
				break;
			}
		}	
		
		tmpParticles[i].centerX = particles[index].centerX;
		tmpParticles[i].centerY = particles[index].centerY;
		tmpParticles[i].velocityX = particles[index].velocityX;
		tmpParticles[i].velocityY = particles[index].velocityY;
		tmpParticles[i].hx = particles[index].hx;
		tmpParticles[i].hy = particles[index].hy;

		// tra'o bo phat sinh so ngau nhien cu ==> de mang lai tinh dda da.ng cho viec phat sinh ngau nhien
		tmpParticles[i].gaussianGenX = particles[i].gaussianGenX;
		tmpParticles[i].gaussianGenY = particles[i].gaussianGenY;
		tmpParticles[i].gaussianGenVelocityX = particles[i].gaussianGenVelocityX;
		tmpParticles[i].gaussianGenVelocityY = particles[i].gaussianGenVelocityY;
		tmpParticles[i].gaussianGenHx = particles[index].gaussianGenHx;
		tmpParticles[i].gaussianGenHy = particles[index].gaussianGenHy;

		//text.Format(L"%4d   ", index);	f.WriteString(text);		
	}
	//f.Close();
	
	// ---- release mem ----
	delete[] cummulativeSumOfWeights;

	//for(int i = 0; i < numParticles; i++) {
	//	delete[] particles[i].colorDistribution;
	//}
	delete[] particles;

	// ----
	particles = tmpParticles;
}

// ================================ PRIVATE ========================================
double ColorBasedPF::distancePitago(int x, int y) { // a
	return sqrt(double(x*x + y*y));
}

double ColorBasedPF::kernel(int x1, int x2, double bandwidthPatch) {
	double r = distancePitago(x1, x2) / bandwidthPatch;
	if (r < 1) return 1 - r*r;
	else return 0;
}

double* ColorBasedPF::calculateRGBHistogram(IplImage* img, CvPoint center, int hx, int hy) { 
	int channels = img->nChannels;		int step = img->widthStep;	int width = img->width;		int height = img->height;
	const uchar* imgData = (uchar *)img->imageData;

	int startX = center.x - hx;		//if(startX < 0) startX = 0;
	int endX = center.x + hx;		//if(endX > width) endX = width;
	int startY = center.y - hy;		//if(startY < 0) startY = 0;
	int endY = center.y + hy;		//if(endY > height) endY = height;
	double bandwidthOfPatch = distancePitago(hx, hy);

	// init histogram bins to all zero
	double* histogram = new double[NUM_BINS_RGB]; // RGB: 8x8x8
	for (int i = 0; i < NUM_BINS_RGB; i++) histogram[i] = 0;

	//CStdioFile f;	f.Open(L"objHist.txt", CFile::modeCreate | CFile::modeWrite);
	//CString text;

	for (int i = startY; i < endY; i++)
		for (int j = startX; j < endX; j++) {
			int pos = i*step + j*channels;
			int binIndexR = imgData[pos] / 32; 	// 256 / 8 = 32
			int binIndexG = imgData[pos + 1] / 32;
			int binIndexB = imgData[pos + 2] / 32;
			histogram[binIndexR*64 + binIndexG*8 + binIndexB] += kernel(j - center.x, i - center.y, bandwidthOfPatch);
		}
			
	// normalize histogram
	double sumOfHistogramValue = 0;
	for (int i = 0; i < NUM_BINS_RGB; i++) sumOfHistogramValue += histogram[i];

	for (int i = 0; i < NUM_BINS_RGB; i++) {
		histogram[i] /= sumOfHistogramValue;
		
		//text.Format(L"%4.8f \n", histogram[i]);	f.WriteString(text);
	}
	//f.Close();

	return histogram;
}

double ColorBasedPF::bhattacharyyaCoefficient(double* sampleColorDistribution, double* objColorDistribution) {
	double result = 0;
	for (int i = 0; i < NUM_BINS_RGB; i++) 
		result += sqrt(sampleColorDistribution[i]*objColorDistribution[i]);
	return result;
}

double ColorBasedPF::weight(double bhaCoeff, double sigma) {
	double factor = 1.0/(sqrt(2*CV_PI)*sigma);
	return factor * exp( - (1-bhaCoeff) / (2*sigma*sigma) );
}

void ColorBasedPF::normalizeWeights(EllipseParticle* particles) {
	double sumWeights = 0;
	for (int i = 0; i < numParticles; i++) sumWeights += particles[i].weight;
	for (int i = 0; i < numParticles; i++) particles[i].weight /= sumWeights;
}

void ColorBasedPF::drawMeanState(IplImage* currFrame, CvRect meanState) {
	CvBox2D box;
	box.center = cvPoint2D32f(meanState.x*1.0, meanState.y*1.0);
	box.size = cvSize2D32f(double(meanState.width*2+1), double(meanState.height*2+1));	
	box.angle = 90;
	cvEllipseBox(currFrame, box, CV_RGB(0,255,0), 2);
}

/*bool ColorBasedPF::isFoundObj(EllipseParticle* particles) {
	//	tinh trung binh, phuong sai cac Bhattacharyya coefficient cua cac ellipse particles (eq. 11 va 12) 
	//	==> de kiem soat xem nhung particles co track theo duoc obj hay ko, neu ko thoa dieu kien track (eq. 13) thi tro ve init-mode

	const double FRACTION_F = 0.1;
	
	// calculate the mean of Bhattacharyya coefficients
	double meanBhaCoeff = 0;
	for (int i = 0; i < numParticles; i++) meanBhaCoeff += particles[i].bhattacharyyaCoefficient;
	meanBhaCoeff /= numParticles;

	// calculate the standard deviation
	double stdDeviation = 0;
	for (int i = 0; i < numParticles; i++) stdDeviation += pow(particles[i].bhattacharyyaCoefficient - meanBhaCoeff, 2.0);
	stdDeviation = sqrt(stdDeviation / numParticles);

	// check the Appearance rule
	int countSatisfiedParticle = 0;
	for (int i = 0; i < numParticles; i++) 
		if (particles[i].bhattacharyyaCoefficient > meanBhaCoeff + 2*stdDeviation)
			countSatisfiedParticle++;

	if (countSatisfiedParticle*1.0 / numParticles > FRACTION_F) // satisfy
		return true;
	else
		return false;
}*/

// ------------------------ for testing ------------------------------

void ColorBasedPF::drawAllParticles(IplImage* currFrame) {
	CvBox2D box; 
	for(int i = 0; i < numParticles; i++){		
		box.center = cvPoint2D32f(particles[i].centerX * 1.0, particles[i].centerY * 1.0);
		box.size = cvSize2D32f(double(particles[i].hx * 2 + 1), double(particles[i].hy * 2 + 1));
		box.angle = 90;
		cvEllipseBox(currFrame, box, CV_RGB(255,0,0));

		/*cvRectangle(currFrame, 
			cvPoint(particles[i].centerX - 1, particles[i].centerY - 1), cvPoint(particles[i].centerX + 1, particles[i].centerY + 1), 
			CV_RGB(255,0,0), CV_FILLED);*/
	}
}

void ColorBasedPF::writeInfoToFile() {
	CStdioFile f;	f.Open(L"particles.txt", CFile::modeCreate | CFile::modeWrite);
	CString text;

	f.WriteString(L"x        y        hx        hy        weight      \n");
	for (int i = 0; i < numParticles; i++) {
		text.Format(L"%3d    ", particles[i].centerX);						f.WriteString(text);
		text.Format(L"%3d    ", particles[i].centerY);						f.WriteString(text);
		text.Format(L"%3d    ", particles[i].hx);							f.WriteString(text);
		text.Format(L"%3d    ", particles[i].hy);							f.WriteString(text);
		text.Format(L"%4.10f \n", particles[i].weight);						f.WriteString(text);
	}
	f.Close();
}
