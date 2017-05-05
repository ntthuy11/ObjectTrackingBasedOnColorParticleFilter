#include "StdAfx.h"
#include "MeanShiftEMScale.h"

MeanShiftEMScale::MeanShiftEMScale(void) { }

MeanShiftEMScale::~MeanShiftEMScale(void) {
	delete[] objHistogram;
}

void MeanShiftEMScale::setParamsAndInit(int objHx, int objHy, int winWidth, int winHeight, double epsilon, CvPoint initPosition, IplImage* initFrame) { 
	this->halfWinWidth = winWidth/2;
	this->halfWinHeight = winHeight/2;
	this->epsilon = epsilon;

	this->currLocation = cvPoint(initPosition.x, initPosition.y);
	this->newLocation = cvPoint(-1, -1);
	this->currVariance = cvPoint(objHx, objHy);
	this->newVariance = cvPoint(-1, -1);

	this->searchBandwidth = distancePitago(halfWinWidth, halfWinHeight);

	objHistogram = calculateRGBHistogram(initFrame, initPosition, objHx, objHy);
}

// --------------------

void MeanShiftEMScale::process(IplImage* currFrame) {

	while (true) {
		double* histogramOfCurrPosition = 0;
		int w = halfWinWidth, h = halfWinHeight;

		// loai tru truong hop cua so tim kiem vuot ra bien
		if (currLocation.x - halfWinWidth < 0)						{ w = currLocation.x; }
		if (currLocation.x + halfWinWidth >= currFrame->width)		{ w = currFrame->width - currLocation.x - 1; }
		if (currLocation.y - halfWinHeight < 0)						{ h = currLocation.y; }
		if (currLocation.y + halfWinHeight >= currFrame->height)	{ h = currFrame->height - currLocation.y - 1; }		 

		//
		histogramOfCurrPosition = calculateRGBHistogram(currFrame, currLocation, w, h);
		double* qs = calculateQs(currFrame, currLocation, histogramOfCurrPosition, w, h);
		newLocation = calculateNewLocationOfTarget(currFrame, currLocation, w, h, qs);
		newVariance = calculateNewVariance(currFrame, currLocation, w, h, qs);

		delete[] histogramOfCurrPosition;
		delete[] qs;

		// kiem tra xem vi tri moi tim duoc co gan vi tri cu khong
		if (isFoundObj(newLocation, currLocation, newVariance, currVariance, epsilon)) { // neu vi tri moi gan vi tri cu trong 1 khoang epsilon nao do thi dung
			currLocation = cvPoint(newLocation.x, newLocation.y);
			currVariance = cvPoint(newVariance.x, newVariance.y);
			break;
		} else {
			currLocation = cvPoint(newLocation.x, newLocation.y);
			currVariance = cvPoint(newVariance.x, newVariance.y);
		}
	}

	drawTracker(currFrame, currLocation, newVariance);
}

double* MeanShiftEMScale::calculateQs(IplImage* img, CvPoint center, double* targetCandidateHistogram, int hx, int hy) {
	int startX = center.x - hx;		int endX = center.x + hx;
	int startY = center.y - hy;		int endY = center.y + hy;

	double* weights = calculateWeights(img, center, targetCandidateHistogram, hx, hy);

	// calculate normalization factor
	int wIndex = 0;
	double normalFactor = 0;
	for (int i = startY; i < endY; i++)
		for (int j = startX; j < endX; j++) {			
			normalFactor += weights[wIndex]*normalKernel(cvPoint(j, i), center, this->currVariance.x, this->currVariance.y);
			wIndex++;
		}
		
	// calculate q(s)
	int qIndex = 0;
	double* qs = new double[4*hx*hy];

	for (int i = startY; i < endY; i++)
		for (int j = startX; j < endX; j++) {
			qs[qIndex] = (weights[qIndex]*normalKernel(cvPoint(j, i), center, this->currVariance.x, this->currVariance.y)) / normalFactor;
			qIndex++;
		}

	//
	delete[] weights;

	//
	return qs;
}

CvPoint MeanShiftEMScale::calculateNewLocationOfTarget(IplImage* img, CvPoint oldLocationOfTarget, int hx, int hy, double* qs) {
	int startX = oldLocationOfTarget.x - hx;		int endX = oldLocationOfTarget.x + hx; 
	int startY = oldLocationOfTarget.y - hy;		int endY = oldLocationOfTarget.y + hy;

	double newX = -1, newY = -1;
	int index = 0;
	for (int i = startY; i < endY; i++)
		for (int j = startX; j < endX; j++) {
			newX += qs[index]*j;
			newY += qs[index]*i;
			index++;
		}

	return cvPoint(int(newX), int(newY));
}

CvPoint MeanShiftEMScale::calculateNewVariance(IplImage* img, CvPoint oldLocationOfTarget, int hx, int hy, double* qs) {
	const double beta = 1.2;
	
	//
	int startX = oldLocationOfTarget.x - hx;		int endX = oldLocationOfTarget.x + hx; 
	int startY = oldLocationOfTarget.y - hy;		int endY = oldLocationOfTarget.y + hy;

	//
	double newHxHx = -1, newHyHy = -1, newHxHy = -1;
	int index = 0;
	for (int i = startY; i < endY; i++)
		for (int j = startX; j < endX; j++) {
			newHxHx += qs[index] * pow(j-oldLocationOfTarget.x, 2.0);
			newHyHy += qs[index] * pow(i-oldLocationOfTarget.y, 2.0);
			index++;
		}

	return cvPoint( int(sqrt(beta*newHxHx)), int(sqrt(beta*newHyHy)) );
}

bool MeanShiftEMScale::isFoundObj(CvPoint newLocation, CvPoint oldLocation, CvPoint newVariance, CvPoint currVariance, double e) {
	return (distancePitago(newLocation.x - oldLocation.x, newLocation.y - oldLocation.y) < e) 
		&& (newVariance.x == currVariance.x) && (newVariance.y == currVariance.y);
}

void MeanShiftEMScale::drawTracker(IplImage* currFrame, CvPoint trackerCenter, CvPoint trackerVariance) {
	CvBox2D box;
	box.center = cvPoint2D32f(trackerCenter.x*1.0, trackerCenter.y*1.0);
	box.size = cvSize2D32f(trackerVariance.x*1.0, trackerVariance.y*1.0);	
	box.angle = 90;
	cvEllipseBox(currFrame, box, CV_RGB(0,255,0), 2);
}

// ================================ PRIVATE ========================================

double MeanShiftEMScale::distancePitago(int x, int y) { 
	return sqrt(double(x*x + y*y));
}

double MeanShiftEMScale::kernel(int x1, int x2, double bandwidthPatch) {
	double r = distancePitago(x1, x2) / bandwidthPatch;
	if (r < 1) return 1 - r*r;
	else return 0;
}

double MeanShiftEMScale::normalKernel(CvPoint y, CvPoint x, int hx, int hy) {
	double a = distancePitago( int((y.x - x.x)*1.0 / hx), int((y.y - x.y)*1.0 / hy) );
	return (1.0/sqrt(2*CV_PI)) * exp(a);
}

double* MeanShiftEMScale::calculateRGBHistogram(IplImage* img, CvPoint center, int hx, int hy) { 
	int channels = img->nChannels;		int step = img->widthStep;
	const uchar* imgData = (uchar *)img->imageData;

	int startX = center.x - hx;		int endX = center.x + hx;
	int startY = center.y - hy;		int endY = center.y + hy;
	double histogramBandwidth = distancePitago(hx, hy);

	// init histogram bins to all zero
	double* histogram = new double[NUM_BINS_RGB]; // RGB: 8x8x8
	for (int i = 0; i < NUM_BINS_RGB; i++) histogram[i] = 0;

	for (int i = startY; i < endY; i++)
		for (int j = startX; j < endX; j++) {
			int jTranslated = j - center.x;
			int iTranslated = i - center.y;
			int binIndexR = imgData[i*step + j*channels] / 32;
			int binIndexG = imgData[i*step + j*channels + 1] / 32;
			int binIndexB = imgData[i*step + j*channels + 2] / 32;
			histogram[binIndexR*64 + binIndexG*8 + binIndexB] += kernel(jTranslated, iTranslated, histogramBandwidth);
		}
	
	// normalize histogram
	double sumOfHistogramValue = 0;
	for (int i = 0; i < NUM_BINS_RGB; i++) sumOfHistogramValue += histogram[i];
	for (int i = 0; i < NUM_BINS_RGB; i++) histogram[i] /= sumOfHistogramValue;

	return histogram;
}

double* MeanShiftEMScale::calculateWeights(IplImage* img, CvPoint center, double* targetCandidateHistogram, int hx, int hy) { // center = locationOfTarget
	int channels = img->nChannels;	int step = img->widthStep;
	const uchar* imgData = (uchar *)img->imageData;

	int startX = center.x - hx;		int endX = center.x + hx;
	int startY = center.y - hy;		int endY = center.y + hy;

	//
	int wIndex = 0;
	int numWeights = 4*hx*hy;
	double* weights = new double[numWeights]; 

	for (int i = startY; i < endY; i++)
		for (int j = startX; j < endX; j++) {
			int ii = i*step + j*channels;
			int binIndexR = imgData[ii] / 32;
			int binIndexG = imgData[ii + 1] / 32;
			int binIndexB = imgData[ii + 2] / 32;
			int binIndex = binIndexR*64 + binIndexG*8 + binIndexB;
			
			weights[wIndex] = sqrt(objHistogram[binIndex] / targetCandidateHistogram[binIndex]);
			wIndex++;
		}
			
	// normalize
	double sumOfWeightValues = 0;
	for (int i = 0; i < numWeights; i++) sumOfWeightValues += weights[i];
	for (int i = 0; i < numWeights; i++) weights[i] /= sumOfWeightValues;

	return weights;
}

/*double MeanShiftEMScale::calculateNormalKernel(CvPoint x, CvPoint mean, CvMat* vMatrix) {
	
	// tinh 2PI*sqrt(det(vMatrix)) 
	double denominator = 2*CV_PI*sqrt(cvDet(vMatrix));

	// tinh exp(...)
	int x1_m1 = x.x - mean.x;
	int x2_m2 = x.y - mean.y;
	
	CvMat* inversionOfVMatrix = cvCreateMat(2, 2, CV_64FC1); // cvMat CHI co khai bao voi CV_64FC1 thi moi su dung duoc cvDet, cvInvert,...
	cvInvert(vMatrix, inversionOfVMatrix);
	double* inversionOfVMatrixData = inversionOfVMatrix->data.db;

	double numerator = exp( -0.5 * (
			(x1_m1*inversionOfVMatrixData[0] + x2_m2*inversionOfVMatrixData[2])*x1_m1
		  + (x1_m1*inversionOfVMatrixData[1] + x2_m2*inversionOfVMatrixData[3])*x2_m2
		));

	//
	cvReleaseMat(&inversionOfVMatrix);

	//
	return numerator/denominator;
}*/
