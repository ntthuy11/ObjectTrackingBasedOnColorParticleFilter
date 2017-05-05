/*  IMPLEMENTATION BASED ON
	Katja Nummiaro, Esther Koller-Meier, and Luc Van Gool, "A Color-Based Particle Filter,"
	in Proceedings of the 1st International Workshop on Generative-Model-Based Vision (GMBV ’02), 
	Copenhagen, Denmark, June 2002
*/

#pragma once

#include "cv.h"
#include "highgui.h"
#include "UniformGen.h"
#include "math.h"

#include "ColorBasedPF.h"

/*
#define NUM_BINS_RGB	(512)

typedef struct EllipseParticle {
	int centerX;
	int centerY;
	int velocityX;
	int velocityY;

	double weight;

	CGaussianGen gaussianGenX; // moi particle se co 1 bo^. phat sinh so ngau nhien rieng biet & ko thay doi bo^. phat sinh nay theo thoi gian
	CGaussianGen gaussianGenY; 
	CGaussianGen gaussianGenVelocityX;
	CGaussianGen gaussianGenVelocityY;
	CGaussianGen gaussianGenHx;
	CGaussianGen gaussianGenHy;
} EllipseParticle;
*/
// -----------------------------

class ColorBasedPF_xyVxVy
{
public:
	ColorBasedPF_xyVxVy(void);
public:
	~ColorBasedPF_xyVxVy(void);

	EllipseParticle* particles;	
	double* objHistogram;

	int numParticles, hx, hy;
	CvPoint expectedPositionOfObj;
	double positionSigma, velocitySigma;

	// -----------------------------------

	void process(IplImage* frame);					  // main function

	void setParamsAndInit(int numParticles,			  // 1. calculate the object's histogram
		int hx, int hy, CvPoint expectedPositionOfObj, 
		double positionSigma, double velocitySigma, 
		IplImage* initFrame);
	
	void initParticles();							  // 2. place particles strategically at positions
													  //    where the target is expected to appear

	void predict(int imgWidth, int imgHeight);		  // 3. predict the next position of the object by 
													  //    propagating each particles by a linear 
													  //    stochastic differential equation

	void calculateParticlesInfo(IplImage* currFrame); // 4. update the color distributions of particles

	CvPoint estimateMeanState();					  // 5. estimate the mean state

	void resamplingCBPF();							  // 6. resample N particles

private:
	double distancePitago(int x, int y);
	double kernel(int x1, int x2, double bandwidthPatch);
	double* calculateRGBHistogram(IplImage* img, CvPoint center, int hx, int hy);	
	double bhattacharyyaCoefficient(double* sampleColorDistribution, double* objColorDistribution);
	double weight(double bhaCoeff, double sigma);	
	void normalizeWeights(EllipseParticle* particles);	
	void drawMeanState(IplImage* currFrame, CvPoint p);

	bool isFoundObj(EllipseParticle* particles);

	// for testing
	void drawAllParticles(IplImage* currFrame);
	void writeInfoToFile();
};
