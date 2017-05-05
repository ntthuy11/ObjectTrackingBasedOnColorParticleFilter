/*  IMPLEMENTATION BASED ON
	Zoran Zivkovic, Ben Kröse, "An EM-Like algorithm for color-histogram-based object tracking", 
	in Proc. IEEE Conference Computer Vision Pattern Recognition, 2004
*/

#pragma once

#include "cv.h"
#include "highgui.h"
#include "UniformGen.h"
#include "math.h"

#define NUM_BINS_RGB	(512)

class MeanShiftEMScale
{
public:
	MeanShiftEMScale(void);
public:
	~MeanShiftEMScale(void);

	void process(IplImage* currFrame);

	void setParamsAndInit(int objHx, int objHy, int winWidth, int winHeight, double epsilon, CvPoint initPosition, IplImage* initFrame);
	double* calculateQs(IplImage* img, CvPoint center, double* targetCandidateHistogram, int hx, int hy);
	CvPoint calculateNewLocationOfTarget(IplImage* img, CvPoint oldLocationOfTarget, int hx, int hy, double* qs);
	CvPoint calculateNewVariance(IplImage* img, CvPoint oldLocationOfTarget, int hx, int hy, double* qs);

	bool isFoundObj(CvPoint newLocation, CvPoint oldLocation, CvPoint newVariance, CvPoint currVariance, double e);
	void drawTracker(IplImage* currFrame, CvPoint trackerCenter, CvPoint trackerVariance);

private:
	double* objHistogram;
	int halfWinWidth, halfWinHeight;
	double epsilon, searchBandwidth;
	CvPoint currLocation, newLocation, currVariance, newVariance;

	double distancePitago(int x, int y);
	double kernel(int x1, int x2, double bandwidthPatch);
	double* calculateRGBHistogram(IplImage* img, CvPoint center, int hx, int hy);
	double* calculateWeights(IplImage* img, CvPoint center, double* targetCandidateHistogram, int hx, int hy);
	
	//double calculateNormalKernel(CvPoint x, CvPoint mean, CvMat* vMatrix);
	double normalKernel(CvPoint y, CvPoint x, int hx, int hy);
};
