#pragma once

#include "cv.h"
#include "highgui.h"
#include "UniformGen.h"

typedef struct PointParticle {
	int x;
	int y;
	int intensity; // intensity chinh la weight chua duoc chuan hoa
	double weight; // voi bai toan nay thi weight chi duoc dung de mo ta, chu ko su dung (su dung intensity la chinh)

	CGaussianGen gaussianGenX; // moi particle se co 1 bo^. phat sinh so ngau nhien rieng biet & ko thay doi bo^. phat sinh nay theo thoi gian
	CGaussianGen gaussianGenY;	
} PointParticle;

// -----------------------------

class GenericParticleFilter
{
public:
	GenericParticleFilter(IplImage* frame, int numParticles, double sigma);
public:
	~GenericParticleFilter(void);

	PointParticle* particles;
	int numParticles;
	double sigma;
	IplImage *previousFrame;

	IplImage* process(IplImage* currFrame);
	void initParticles();
	void addNoiseToParticles();
	void getIntensityForParticles(IplImage* observedFrame);
	void resampling();
};
