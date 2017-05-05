/*  IMPLEMENTATION BASED ON
	Katja Nummiaro, Esther Koller-Meier, and Luc Van Gool, "A Color-Based Particle Filter,"
	in Proceedings of the 1st International Workshop on Generative-Model-Based Vision (GMBV ’02), 
	Copenhagen, Denmark, June 2002


	** dynamic model: s = {x,y,vx,vy,hx,hy}
*/

#pragma once

#include "cv.h"
#include "highgui.h"
#include "UniformGen.h"
#include "math.h"

#define NUM_BINS_RGB	(512)

typedef struct EllipseParticle {
	int centerX;
	int centerY;
	int velocityX;
	int velocityY;
	int hx;
	int hy;
	double weight;

	CGaussianGen gaussianGenX; // moi particle se co 1 bo^. phat sinh so ngau nhien rieng biet & ko thay doi bo^. phat sinh nay theo thoi gian
	CGaussianGen gaussianGenY; 
	CGaussianGen gaussianGenVelocityX;
	CGaussianGen gaussianGenVelocityY;
	CGaussianGen gaussianGenHx;
	CGaussianGen gaussianGenHy;
} EllipseParticle;

// -----------------------------

class ColorBasedPF
{
public:
	ColorBasedPF(void);
public:
	~ColorBasedPF(void);

	EllipseParticle* particles;	
	double* objHistogram;

	int numParticles, hx, hy;
	CvPoint expectedPositionOfObj;
	double positionSigma, velocitySigma, sizeSigma;

	// -----------------------------------

	void setParamsAndInit(int numParticles, int hx, int hy, CvPoint expectedPositionOfObj, double positionSigma, double velocitySigma, 
		double sizeSigma, IplImage* initFrame);

	void process(IplImage* frame);
	void initParticles();
	void predict(int imgWidth, int imgHeight);
	void calculateParticlesInfo(IplImage* currFrame);
	CvRect estimateMeanState();
	void resamplingCBPF();

private:
	double distancePitago(int x, int y);
	double kernel(int x1, int x2, double bandwidthPatch);
	double* calculateRGBHistogram(IplImage* img, CvPoint center, int hx, int hy);	
	double bhattacharyyaCoefficient(double* sampleColorDistribution, double* objColorDistribution);
	double weight(double bhaCoeff, double sigma);	
	void normalizeWeights(EllipseParticle* particles);	
	void drawMeanState(IplImage* currFrame, CvRect p);

	bool isFoundObj(EllipseParticle* particles);

	// for testing
	void drawAllParticles(IplImage* currFrame);
	void writeInfoToFile();
};
