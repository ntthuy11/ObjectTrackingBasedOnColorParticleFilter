/*  IMPLEMENTATION BASED ON
	D.Comaniciu, V.Ramesh, and P.Meer, "Real-time tracking of non-rigid objects using mean shift", 
	in Proc. IEEE Conf. on Computer Vision and Pattern Recognition, 2:142-149, 2000
*/

#pragma once

#include "cv.h"
#include "highgui.h"
#include "UniformGen.h"
#include "math.h"

#define NUM_BINS_RGB	(512)

class MeanShift
{
public:
	MeanShift(void);
public:
	~MeanShift(void);

	// --------------------------------
	void setParamsAndInit(int objHx, int objHy, int winWidth, int winHeight, double epsilon, CvPoint initPosition, IplImage* initFrame);

	void process(IplImage* currFrame);	
	CvPoint calculateNewLocationOfTarget(IplImage* img, CvPoint oldLocationOfTarget, double* targetCandidateHistogram, int hx, int hy);
	bool isFoundObj(CvPoint newLocation, CvPoint oldLocation, double e);
	void drawTracker(IplImage* currFrame, CvPoint tracker);

private:
	double* objHistogram;
	int objHx, objHy, halfWinWidth, halfWinHeight;
	double epsilon, searchBandwidth;
	CvPoint currLocation, newLocation;

	double distancePitago(int x, int y);
	double kernel(int x1, int x2, double bandwidthPatch);
	double normalKernel(CvPoint y, CvPoint x, double bandwidth);
	double* calculateRGBHistogram(IplImage* img, CvPoint center, int hx, int hy);
	double* calculateWeights(IplImage* img, CvPoint center, double* targetCandidateHistogram, int hx, int hy);
};
