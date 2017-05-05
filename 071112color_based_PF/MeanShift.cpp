#include "StdAfx.h"
#include "MeanShift.h"

MeanShift::MeanShift(void) { }

MeanShift::~MeanShift(void) {
	delete[] objHistogram;
}

void MeanShift::setParamsAndInit(int objHx, int objHy, int winWidth, int winHeight, double epsilon, CvPoint initPosition, IplImage* initFrame) { 
	this->objHx = objHx;
	this->objHy = objHy;
	this->halfWinWidth = winWidth/2;
	this->halfWinHeight = winHeight/2;
	this->epsilon = epsilon;

	this->currLocation = cvPoint(initPosition.x, initPosition.y);
	this->newLocation = cvPoint(-1, -1);

	this->searchBandwidth = distancePitago(halfWinWidth, halfWinHeight);

	objHistogram = calculateRGBHistogram(initFrame, initPosition, objHx, objHy);
}

// --------------------

void MeanShift::process(IplImage* currFrame) {

	while (true) {
		double* histogramOfCurrPosition = 0;
		int w = halfWinWidth, h = halfWinHeight;

		// loai tru truong hop cua so tim kiem vuot ra bien
		if (currLocation.x - halfWinWidth < 0)						{ w = currLocation.x; }
		if (currLocation.x + halfWinWidth >= currFrame->width)		{ w = currFrame->width - currLocation.x - 1; }
		if (currLocation.y - halfWinHeight < 0)						{ h = currLocation.y; }
		if (currLocation.y + halfWinHeight >= currFrame->height)	{ h = currFrame->height - currLocation.y - 1; }		 

		histogramOfCurrPosition = calculateRGBHistogram(currFrame, currLocation, w, h);
		newLocation = calculateNewLocationOfTarget(currFrame, currLocation, histogramOfCurrPosition, w, h);
		delete[] histogramOfCurrPosition;

		// kiem tra xem vi tri moi tim duoc co gan vi tri cu khong
		if (isFoundObj(newLocation, currLocation, epsilon)) { // neu vi tri moi gan vi tri cu trong 1 khoang epsilon nao do thi dung
			currLocation = cvPoint(newLocation.x, newLocation.y);
			break;
		} else {
			currLocation = cvPoint(newLocation.x, newLocation.y);
		}
	}

	drawTracker(currFrame, currLocation);
}

CvPoint MeanShift::calculateNewLocationOfTarget(IplImage* img, CvPoint oldLocationOfTarget, double* targetCandidateHistogram, int hx, int hy) {
	int startX = oldLocationOfTarget.x - hx;		int endX = oldLocationOfTarget.x + hx; 
	int startY = oldLocationOfTarget.y - hy;		int endY = oldLocationOfTarget.y + hy;

	double* weights = calculateWeights(img, oldLocationOfTarget, targetCandidateHistogram, hx, hy); 
	double numeratorX1 = 0, numeratorX2 = 0, denominator = 0;
	int index = 0;	

	for (int i = startY; i < endY; i++)
		for (int j = startX; j < endX; j++) {
			double wg = weights[index] * normalKernel(oldLocationOfTarget, cvPoint(j, i), searchBandwidth);
			numeratorX1 += i * wg;
			numeratorX2 += j * wg;
			denominator += wg;
			index++;
		}

	delete[] weights;

	return cvPoint(int(numeratorX2 / denominator), int(numeratorX1 / denominator));
}

bool MeanShift::isFoundObj(CvPoint newLocation, CvPoint oldLocation, double e) {
	return (distancePitago(newLocation.x - oldLocation.x, newLocation.y - oldLocation.y) < e);
}

void MeanShift::drawTracker(IplImage* currFrame, CvPoint tracker) {
	CvBox2D box;
	box.center = cvPoint2D32f(tracker.x*1.0, tracker.y*1.0);
	box.size = cvSize2D32f(double(objHy*2), double(objHy*2));	
	box.angle = 90;
	cvEllipseBox(currFrame, box, CV_RGB(0,255,0), 2);
}

// ================================ PRIVATE ========================================

double MeanShift::distancePitago(int x, int y) { 
	return sqrt(double(x*x + y*y));
}

double MeanShift::kernel(int x1, int x2, double bandwidthPatch) {
	double r = distancePitago(x1, x2) / bandwidthPatch;
	if (r < 1) return 1 - r*r;
	else return 0;
}

double MeanShift::normalKernel(CvPoint y, CvPoint x, double bandwidth) {
	double a = distancePitago( int((y.x - x.x) / bandwidth), int((y.y - x.y) / bandwidth) );
	return (1.0/sqrt(2*CV_PI)) * exp(a);
}

double* MeanShift::calculateRGBHistogram(IplImage* img, CvPoint center, int hx, int hy) { 
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

double* MeanShift::calculateWeights(IplImage* img, CvPoint center, double* targetCandidateHistogram, int hx, int hy) { // center = locationOfTarget
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
