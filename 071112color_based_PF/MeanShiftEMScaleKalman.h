#pragma once

#include "cv.h"
#include "highgui.h"

#define BETA			(1.2)
#define N_INTERATION	(6)
#define LAMBDA			(0.2) // OBSERVATION MODEL: P ~ EXP(-0.5*(1-RHO)/LAMBDA^2)
#define N_BINS			(8)
#define SIGMA			(2.5)
#define N_STATES		(5)
#define FCLIP			(0.005)


typedef struct GaussianKernel {
	CvMat* rK; // 1 x n					// CV_32FC1
	CvMat* rX;							// int
	CvMat* rY;							// int
	CvMat* rXnormalized;				// CV_32FC1
	CvMat* rYnormalized;				// CV_32FC1
	CvMat* sigma; // [sigmaX sigmaY]	// CV_32FC1
	CvMat* size; // [width height]		// int
} GaussianKernel;


typedef struct Histogram {
	CvMat* data;
	CvMat* originH;
} Histogram;


//  ------------------------

class MeanShiftEMScaleKalman
{
public:
	MeanShiftEMScaleKalman(void);
	~MeanShiftEMScaleKalman(void);

	void setParamsAndInit(int objHx, int objHy, int winWidth, int winHeight, double epsilon, CvPoint initPosition, IplImage* initFrame);
	void process(IplImage* currFrame);

private:
	CvMat* stateVector;
	CvMat* pkMat;
	GaussianKernel gaussKernel;
	Histogram objHistogram;

	void calculateCovarianceMatrix(int hx, int hy, CvPoint center, CvMat* covMat);
	void generateGaussianKernel(CvMat* covMat);
	void getAffineRegion(IplImage* srcImg, CvPoint mean, CvMat* covMat, 
		CvMat* imgROIchannelR, CvMat* imgROIchannelG, CvMat* imgROIchannelB, CvMat* rX, CvMat* rY);
	void getHistogram(CvMat* imgROIchannelR, CvMat* imgROIchannelG, CvMat* imgROIchannelB, Histogram *histogram);
	void convertMeanAndVarianceToStateVector(CvPoint mean, CvMat *covMat, CvMat* stateVector);
	void convertStateVectorToMeanAndVariance(CvMat* stateVector, CvPoint* mean, CvMat *covMat);
	void EMShift(IplImage* srcImg, CvPoint mean, CvMat* covMat, 
		CvPoint* newMean, CvMat *newCovMat);
	void fitQuadraticToBhattacharyaCoeff(IplImage* srcImg, CvMat* stateVector, CvMat* R);
	float calculateBhattacharayyaCoeff(IplImage* srcImg, CvPoint mean, CvMat* covMat);
	void drawTracker(IplImage* srcImg, CvMat* stateVector);

	void interp2(IplImage* srcImg, CvMat* rX, CvMat* rY, CvMat* imgROIr, CvMat* imgROIg, CvMat* imgROIb);

	// matrix methods
	void choleskyDecompose(CvMat* covMat, CvMat* choleskyMat);
	void generateArraysFor3DPlot(int sizX, int sizY, CvMat* xArray, CvMat* yArray);
	void reshapeKernelToOneRowFloat(CvMat* src, CvMat* dst);
	void reshapeKernelToOneRowInt(CvMat* src, CvMat* dst);
	void divideMatrixByANumber(CvMat* src, CvMat* dst, float n);
	void generateIdentityMatrix(CvMat* mat);

	// math methods
	float sqr(float n);
	void generateCircleWithSigmaRadius(CvMat *rCircle);

	// matlab	
	void interp2FromMatlabCMathLibrary(IplImage* srcImg, CvMat* rX, CvMat* rY, CvMat* imgROIr, CvMat* imgROIg, CvMat* imgROIb); // chi dung duoc o Matlab 6.5
	void interp2FromMatlabEngine(IplImage* srcImg, CvMat* rX, CvMat* rY, CvMat* imgROIr, CvMat* imgROIg, CvMat* imgROIb); // chay o Matlab 2007
};
