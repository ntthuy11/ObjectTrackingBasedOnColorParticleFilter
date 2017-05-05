#include "StdAfx.h"
#include "MeanShiftEMScaleKalman.h"
//#include "engine.h"
//#include "matlab.h"

MeanShiftEMScaleKalman::MeanShiftEMScaleKalman(void) { }
MeanShiftEMScaleKalman::~MeanShiftEMScaleKalman(void) { }

void MeanShiftEMScaleKalman::setParamsAndInit(int objHx, int objHy, int winWidth, int winHeight, double epsilon, CvPoint initPosition, IplImage* initFrame) { 
	
	// ------ covariance matrix of obj ------
	CvMat *covMat = cvCreateMat(2, 2, CV_32FC1);
	calculateCovarianceMatrix(objHx, objHy, initPosition, covMat);

	// debug covMat
	//float covMat00 = CV_MAT_ELEM(*covMat, float, 0, 0);		float covMat01 = CV_MAT_ELEM(*covMat, float, 0, 1);	
	//float covMat10 = CV_MAT_ELEM(*covMat, float, 1, 0);		float covMat11 = CV_MAT_ELEM(*covMat, float, 1, 1);	


	// -------- generate Gaussian Kernel of obj ----------
	
	// debug using dummy covMat 
	/*CV_MAT_ELEM(*covMat, float, 0, 0) = 34; CV_MAT_ELEM(*covMat, float, 0, 1) = 2; CV_MAT_ELEM(*covMat, float, 1, 0) = 2;	CV_MAT_ELEM(*covMat, float, 1, 1) = 24;*/

	generateGaussianKernel(covMat); // gaussian kernel duoc generate vao global var


	// -------- generate Affine Region of obj ----------

	// "empty" imgROI
	CvMat* imgROIchannelR = cvCreateMat(1, gaussKernel.rX->width, CV_32FC1); // gaussKernel.rX->height = 1
	CvMat* imgROIchannelG = cvCreateMat(1, gaussKernel.rX->width, CV_32FC1);
	CvMat* imgROIchannelB = cvCreateMat(1, gaussKernel.rX->width, CV_32FC1);

	// newRX, newRY
	CvMat* newRX = cvCreateMat(1, gaussKernel.rX->width, CV_32FC1);
	CvMat* newRY = cvCreateMat(1, gaussKernel.rX->width, CV_32FC1);

	getAffineRegion(initFrame, initPosition, covMat, imgROIchannelR, imgROIchannelG, imgROIchannelB, newRX, newRY);


	// -------- get obj histogram ----------
	getHistogram(imgROIchannelR, imgROIchannelG, imgROIchannelB, &objHistogram);


	// ---------- state vector ----------
	stateVector = cvCreateMat(1, N_STATES, CV_32FC1);
	convertMeanAndVarianceToStateVector(initPosition, covMat, stateVector);


	// ---------- init parameters for DYNAMIC MODEL x = A*x + W	where x is the state vector, W is noise ----------
	/*	A = [ 1 0 0 0 0			// KHONG CAN THIET PHAI KHOI TAO A, W
			  0 1 0 0 0
			  0 0 1 0 0
			  0 0 0 1 0
			  0 0 0 0 1 ]	
		W = [ 9 0 0 0 0
			  0 9 0 0 0
			  0 0 1 0 0
			  0 0 0 1 0
			  0 0 0 0 1 ]	*/

	/*	Pk = [ 10000   0    0     0     0
			    0   10000   0     0     0
			    0     0   10000   0     0
			    0     0     0   10000   0
			    0     0     0     0   10000 ]	*/ // some high initial value for the variance
	pkMat = cvCreateMat(N_STATES, N_STATES, CV_32FC1); // Pk
	generateIdentityMatrix(pkMat);
	CV_MAT_ELEM(*pkMat, float, 0, 0) = CV_MAT_ELEM(*pkMat, float, 1, 1) = CV_MAT_ELEM(*pkMat, float, 2, 2) 
									 = CV_MAT_ELEM(*pkMat, float, 3, 3) = CV_MAT_ELEM(*pkMat, float, 4, 4) = 10000;	
}

void MeanShiftEMScaleKalman::process(IplImage* currFrame) {
	
	// ---------- predict ----------	
	// cvMatMul(identityMat, stateVector, stateVector); // project the state ahead // nhan voi ma tran I (la ma tran A) nen ko can thuc hien
	
	// Pk = A*Pk*A' + W
	/*	W = [ 9 0 0 0 0
			  0 9 0 0 0
			  0 0 1 0 0
			  0 0 0 1 0
			  0 0 0 0 1 ]	*/
	CV_MAT_ELEM(*pkMat, float, 0, 0) += 9;		CV_MAT_ELEM(*pkMat, float, 1, 1) += 9;
	CV_MAT_ELEM(*pkMat, float, 2, 2) += 1;		CV_MAT_ELEM(*pkMat, float, 3, 3) += 1;		CV_MAT_ELEM(*pkMat, float, 4, 4) += 1;


	// ---------- find max ----------
	CvPoint mean = cvPoint(0, 0);
	CvMat* covMat = cvCreateMat(2, 2, CV_32FC1); 
	convertStateVectorToMeanAndVariance(stateVector, &mean, covMat); // prediction as starting point

	for (int i = 0; i < N_INTERATION; i++) 
		EMShift(currFrame, mean, covMat, &mean, covMat);
	
	CvMat* measurementOfStateVector = cvCreateMat(1, N_STATES, CV_32FC1); 
	convertMeanAndVarianceToStateVector(mean, covMat, measurementOfStateVector);


	// ---------- tao measurement noise, voi noise la theo guassian ----------
	CvMat* R = cvCreateMat(1, N_STATES, CV_32FC1); // measurement noise
	fitQuadraticToBhattacharyaCoeff(currFrame, measurementOfStateVector, R); // * sqr(LAMBDA);


	// ---------- compute the Kalman gain ----------
	CvMat* pkR = cvCreateMat(N_STATES, N_STATES, CV_32FC1);
	generateIdentityMatrix(pkR);
	CV_MAT_ELEM(*pkR, float, 0, 0) = CV_MAT_ELEM(*pkMat, float, 0, 0) + CV_MAT_ELEM(*R, float, 0, 0);
	CV_MAT_ELEM(*pkR, float, 1, 1) = CV_MAT_ELEM(*pkMat, float, 1, 1) + CV_MAT_ELEM(*R, float, 0, 1);
	CV_MAT_ELEM(*pkR, float, 2, 2) = CV_MAT_ELEM(*pkMat, float, 2, 2) + CV_MAT_ELEM(*R, float, 0, 2);
	CV_MAT_ELEM(*pkR, float, 3, 3) = CV_MAT_ELEM(*pkMat, float, 3, 3) + CV_MAT_ELEM(*R, float, 0, 3);
	CV_MAT_ELEM(*pkR, float, 4, 4) = CV_MAT_ELEM(*pkMat, float, 4, 4) + CV_MAT_ELEM(*R, float, 0, 4);
	
	CvMat* pkRInvert = cvCreateMat(N_STATES, N_STATES, CV_32FC1);
	cvInvert(pkR, pkRInvert);

	CvMat* KFgain = cvCreateMat(N_STATES, N_STATES, CV_32FC1); // KFgain = Pk * inv(Pk + R)
	cvMatMul(pkMat, pkRInvert, KFgain);


	// ---------- update estimate with measurement ----------

	// calculate (z - s) = (measurementOfStateVector - stateVector)
	CvMat* zMinusS = cvCreateMat(1, N_STATES, CV_32FC1);
	for (int i = 0; i < N_STATES; i++)
		CV_MAT_ELEM(*zMinusS, float, 0, i) = CV_MAT_ELEM(*measurementOfStateVector, float, 0, i) - CV_MAT_ELEM(*stateVector, float, 0, i);

	// transpose (z - s)
	CvMat* zMinusSTranspose = cvCreateMat(N_STATES, 1, CV_32FC1);
	cvTranspose(zMinusS, zMinusSTranspose);

	// calculate KFgain*(z - s)
	CvMat* KF_ZS = cvCreateMat(N_STATES, 1, CV_32FC1);
	cvMatMul(KFgain, zMinusSTranspose, KF_ZS);

	// calculate s = s + KFgain*(z - s)
	for (int i = 0; i < N_STATES; i++)
		CV_MAT_ELEM(*stateVector, float, 0, i) += CV_MAT_ELEM(*KF_ZS, float, i, 0);


	// ---------- update the error covariance ----------
	CvMat* IMinusKFgain = cvCreateMat(N_STATES, N_STATES, CV_32FC1);
	generateIdentityMatrix(IMinusKFgain);
	for (int i = 0; i < N_STATES; i++)
		CV_MAT_ELEM(*IMinusKFgain, float, i, i) = 1 - CV_MAT_ELEM(*KFgain, float, i, i);

	cvMatMul(IMinusKFgain, pkMat, pkMat); // Pk = (I - KFgain) * Pk


	// ---------- draw tracker ----------
	drawTracker(currFrame, stateVector);
}

// ================================ PRIVATE ========================================

void MeanShiftEMScaleKalman::calculateCovarianceMatrix(int hx, int hy, CvPoint center, CvMat* covMat) {
		
	// ----- get all points (vectors) in current window (defined by hx, hy, center) -----
	int vectorsIndex = 0;
	const int startX = center.x - hx;		const int endX = center.x + hx;
	const int startY = center.y - hy;		const int endY = center.y + hy;	

	int numInputVectors = (endX - startX + 1)*(endY - startY + 1);
	CvMat** vectors = new CvMat*[numInputVectors];

	for (int i = startY; i <= endY; i++) 
		for (int j = startX; j <= endX; j++) {
			CvMat *vec = cvCreateMat(2, 1, CV_32FC1);	CV_MAT_ELEM(*vec, float, 0, 0) = (float)j;		CV_MAT_ELEM(*vec, float, 1, 0) = (float)i;
			vectors[vectorsIndex] = vec;
			vectorsIndex++;
		}	
	
	// ----- calculate covMat and Mean -----
	CvMat *mean = cvCreateMat(2, 1, CV_32FC1); // not used
	cvCalcCovarMatrix((const CvArr**)vectors, numInputVectors, covMat, mean, CV_COVAR_NORMAL|CV_COVAR_SCALE); // can vectors kieu float
	
	// debug Mean (must be equals to ObjPosition in the init step)
	//float mean00 = CV_MAT_ELEM(*mean, float, 0, 0);		float mean10 = CV_MAT_ELEM(*mean, float, 1, 0);	

	// ----- release -----
	for (int i = 0; i < numInputVectors; i++) cvReleaseMat(&vectors[i]);
	delete[] vectors;

	cvReleaseMat(&mean);
}

void MeanShiftEMScaleKalman::generateGaussianKernel(CvMat* covMat) {

	// -------------- calculate rCircle 2x101 (tao ra 1 vong tron voi ban kinh la SIGMA) ---------------
	int nPoints = 101;
	CvMat *rCircle = cvCreateMat(2, nPoints, CV_32FC1);		generateCircleWithSigmaRadius(rCircle);


	// -------------- calculate k 2x2, where k' * k = covMat ---------------
	CvMat* k = cvCreateMat(2, 2, CV_32FC1); 
	choleskyDecompose(covMat, k); // k' * k = covMat 

	// debug k
	/*float covMat00 = CV_MAT_ELEM(*covMat, float, 0, 0);		float covMat01 = CV_MAT_ELEM(*covMat, float, 0, 1);	
	float covMat10 = CV_MAT_ELEM(*covMat, float, 1, 0);		float covMat11 = CV_MAT_ELEM(*covMat, float, 1, 1);	
	float k00 = CV_MAT_ELEM(*k, float, 0, 0);		float k01 = CV_MAT_ELEM(*k, float, 0, 1);	
	float k10 = CV_MAT_ELEM(*k, float, 1, 0);		float kt11 = CV_MAT_ELEM(*k, float, 1, 1);	*/

	
	// -------------- calculate rPlot 2x101 (tao ra hinh elip tuong ung voi ma tran k) ---------------
	CvMat* kTranspose = cvCreateMat(2, 2, CV_32FC1); 	cvTranspose(k, kTranspose);
	CvMat* rPlot = cvCreateMat(2, nPoints, CV_32FC1); 	cvMatMul(kTranspose, rCircle, rPlot);

	// debug rPlot
	/*CStdioFile f;	f.Open(L"rPlot.txt", CFile::modeCreate | CFile::modeWrite);	CString text;
	for (int i = 0; i < nPoints; i++) { 
		text.Format(L"%4.4f   ", CV_MAT_ELEM(*rPlot, float, 0, i));		f.WriteString(text);
		text.Format(L"%4.4f \n", CV_MAT_ELEM(*rPlot, float, 1, i));		f.WriteString(text);
	} f.Close();*/


	// -------------- find min & max columns of rPlot ---------------
	CvMat* minColOfPlot = cvCreateMat(2, 1, CV_32FC1); 	CV_MAT_ELEM(*minColOfPlot, float, 0, 0) = .0; 	CV_MAT_ELEM(*minColOfPlot, float, 1, 0) = .0;
	CvMat* maxColOfPlot = cvCreateMat(2, 1, CV_32FC1); 	CV_MAT_ELEM(*maxColOfPlot, float, 0, 0) = .0; 	CV_MAT_ELEM(*maxColOfPlot, float, 1, 0) = .0;
	
	for (int i = 0; i < nPoints; i++) {
		if (   (CV_MAT_ELEM(*minColOfPlot, float, 0, 0) > CV_MAT_ELEM(*rPlot, float, 0, i)) 
			&& (CV_MAT_ELEM(*minColOfPlot, float, 1, 0) > CV_MAT_ELEM(*rPlot, float, 1, i)) ) {
				CV_MAT_ELEM(*minColOfPlot, float, 0, 0) = CV_MAT_ELEM(*rPlot, float, 0, i);
				CV_MAT_ELEM(*minColOfPlot, float, 1, 0) = CV_MAT_ELEM(*rPlot, float, 1, i);
		}
		if (   (CV_MAT_ELEM(*maxColOfPlot, float, 0, 0) < CV_MAT_ELEM(*rPlot, float, 0, i)) 
			&& (CV_MAT_ELEM(*maxColOfPlot, float, 1, 0) < CV_MAT_ELEM(*rPlot, float, 1, i)) ) {
				CV_MAT_ELEM(*maxColOfPlot, float, 0, 0) = CV_MAT_ELEM(*rPlot, float, 0, i);
				CV_MAT_ELEM(*maxColOfPlot, float, 1, 0) = CV_MAT_ELEM(*rPlot, float, 1, i);
		}
	}

	//float* data = minColOfPlot->data.fl;	float a = data[0];	float b = data[1];	// debug minColOfPlot
	//data = maxColOfPlot->data.fl;			float c = data[0];	float d = data[1];	// debug maxColOfPlot


	// -------------- calculate SIGMA, siz --------------
	float sigmaX = CV_MAT_ELEM(*k, float, 0, 0);
	float sigmaY = CV_MAT_ELEM(*k, float, 1, 1);
	int sizX = cvRound( (CV_MAT_ELEM(*maxColOfPlot, float, 0, 0) - CV_MAT_ELEM(*minColOfPlot, float, 0, 0)) / 2 );
	int sizY = cvRound( (CV_MAT_ELEM(*maxColOfPlot, float, 1, 0) - CV_MAT_ELEM(*minColOfPlot, float, 1, 0)) / 2 );


	// -------------- generate arrays for 3D Plot --------------
	int xyArrayHeight = sizY*2 + 1;
	int xyArrayWidth = sizX*2 + 1;
	CvMat* xArray = cvCreateMat(xyArrayHeight, xyArrayWidth, CV_32SC1); 
	CvMat* yArray = cvCreateMat(xyArrayHeight, xyArrayWidth, CV_32SC1); 

	generateArraysFor3DPlot(sizX, sizY, xArray, yArray); // Matlab's meshgrid

	// debug xArray, yArray
	/*CStdioFile f;	f.Open(L"xArray.txt", CFile::modeCreate | CFile::modeWrite);	CString text;
	CStdioFile f0;	f0.Open(L"yArray.txt", CFile::modeCreate | CFile::modeWrite);	CString text0;
	for (int i = 0; i < xyArrayHeight; i++) {
		for (int j = 0; j < xyArrayWidth; j++) {
			text.Format(L"%4d   ", CV_MAT_ELEM(*xArray, int, i, j));		f.WriteString(text);
			text0.Format(L"%4d   ", CV_MAT_ELEM(*yArray, int, i, j));		f0.WriteString(text0);
		} f.WriteString(L"\n"); f0.WriteString(L"\n");
	} f.Close(); f0.Close(); */


	// -------------- calculate the kernel --------------
	CvMat* kernel = cvCreateMat(xyArrayHeight, xyArrayWidth, CV_32FC1); 
	for (int i = 0; i < xyArrayHeight; i++)
		for (int j = 0; j < xyArrayWidth; j++) {
			float disX = sqr( (float) CV_MAT_ELEM(*xArray, int, i, j)) / sqr(sigmaX);
			float disY = sqr( (float) CV_MAT_ELEM(*yArray, int, i, j)) / sqr(sigmaY);
			CV_MAT_ELEM(*kernel, float, i, j) = float( exp( -(disX + disY)/2.0 ) );
		}

	
	// -------------- normalize the kernel --------------
	//cvNormalize(kernel, kernel);

	float sum = 0;
	for (int i = 0; i < xyArrayHeight; i++) for (int j = 0; j < xyArrayWidth; j++) sum += CV_MAT_ELEM(*kernel, float, i, j);
	for (int i = 0; i < xyArrayHeight; i++) for (int j = 0; j < xyArrayWidth; j++) CV_MAT_ELEM(*kernel, float, i, j) /= sum;

	// debug kernel
	CStdioFile f;	f.Open(L"kernel.txt", CFile::modeCreate | CFile::modeWrite);	CString text;
	for (int i = 0; i < xyArrayHeight; i++) {
		for (int j = 0; j < xyArrayWidth; j++) {
			text.Format(L"%4.4f   ", CV_MAT_ELEM(*kernel, float, i, j));		f.WriteString(text);
		} f.WriteString(L"\n");
	} f.Close();


	// -------------- return the Gaussian Kernel --------------
	int oneDimensionArraySize = xyArrayHeight*xyArrayWidth;

	gaussKernel.rK = cvCreateMat(1, xyArrayHeight*xyArrayWidth, CV_32FC1); 
	gaussKernel.rX = cvCreateMat(1, xyArrayHeight*xyArrayWidth, CV_32SC1); 
	gaussKernel.rY = cvCreateMat(1, xyArrayHeight*xyArrayWidth, CV_32SC1); 
	gaussKernel.rXnormalized = cvCreateMat(1, xyArrayHeight*xyArrayWidth, CV_32FC1); 
	gaussKernel.rYnormalized = cvCreateMat(1, xyArrayHeight*xyArrayWidth, CV_32FC1); 
	gaussKernel.sigma = cvCreateMat(1, 2, CV_32FC1); 
	gaussKernel.size = cvCreateMat(1, 2, CV_32SC1); 

	reshapeKernelToOneRowFloat(kernel, gaussKernel.rK); // column-wise
	reshapeKernelToOneRowInt(xArray, gaussKernel.rX);
	reshapeKernelToOneRowInt(yArray, gaussKernel.rY);
	divideMatrixByANumber(gaussKernel.rX, gaussKernel.rXnormalized, sigmaX);
	divideMatrixByANumber(gaussKernel.rY, gaussKernel.rYnormalized, sigmaY);
	
	CV_MAT_ELEM(*gaussKernel.sigma, float, 0, 0) = sigmaX;
	CV_MAT_ELEM(*gaussKernel.sigma, float, 0, 1) = sigmaY;

	CV_MAT_ELEM(*gaussKernel.size, int, 0, 0) = xyArrayHeight;
	CV_MAT_ELEM(*gaussKernel.size, int, 0, 1) = xyArrayWidth;

	// debug gaussKernel
	/*CStdioFile f1;	f1.Open(L"gaussKernel_rX.txt", CFile::modeCreate | CFile::modeWrite);	CString text1;
	for (int i = 0; i < oneDimensionArraySize; i++) {
		text1.Format(L"%4d   ", CV_MAT_ELEM(*gaussKernel.rX, int, 0, i));		f1.WriteString(text1);
		//text1.Format(L"%4.4f   ", CV_MAT_ELEM(*gaussKernel.rK, float, 0, i));		f1.WriteString(text1);
	} f1.Close();*/


	// -------------- release --------------
	cvReleaseMat(&rCircle);	
	cvReleaseMat(&k);
	cvReleaseMat(&kTranspose);
	cvReleaseMat(&rPlot);
	cvReleaseMat(&minColOfPlot);
	cvReleaseMat(&maxColOfPlot);
	cvReleaseMat(&xArray);
	cvReleaseMat(&yArray);
	cvReleaseMat(&kernel);
}

void MeanShiftEMScaleKalman::getAffineRegion(IplImage* srcImg, CvPoint mean, CvMat* covMat, 
											 CvMat* imgROIchannelR, CvMat* imgROIchannelG, CvMat* imgROIchannelB, CvMat* newRx, CvMat* newRy) {
	int kernelWidth = gaussKernel.rX->width;
	
	// debug gaussKernel
	/*CStdioFile f1;	f1.Open(L"gaussKernel.rX.txt", CFile::modeCreate | CFile::modeWrite);	CString text1;
	for (int i = 0; i < kernelWidth; i++) { 
		text1.Format(L"%4d   ", CV_MAT_ELEM(*gaussKernel.rX, int, 0, i));		f1.WriteString(text1);
		//text1.Format(L"%4.4f   ", CV_MAT_ELEM(*gaussKernel.rK, float, 0, i));		f1.WriteString(text1);
	} f1.Close();*/


	// -------- calculate eigenValues & eigenVectors from covariance matrix --------
	CvMat* eigenValues = cvCreateMat(2, 1, CV_32FC1); // vi covMat 2x2 nen eigenValues cung la 2x2
	CvMat* eigenVectors = cvCreateMat(2, 2, CV_32FC1);
	cvEigenVV(covMat, eigenVectors, eigenValues);

	// vi eigenVectors ra dang [0 x; x 0] nen phai chuyen thanh [x 0; 0 x]
	//CV_MAT_ELEM(*eigenVectors, float, 0, 0) = CV_MAT_ELEM(*eigenVectors, float, 0, 1);		CV_MAT_ELEM(*eigenVectors, float, 0, 1) = .0;
	//CV_MAT_ELEM(*eigenVectors, float, 1, 1) = CV_MAT_ELEM(*eigenVectors, float, 1, 0);		CV_MAT_ELEM(*eigenVectors, float, 1, 0) = .0;


	// -------- calculate A --------
	CvMat* m = cvCreateMat(2, 2, CV_32FC1); // m = diag(sqrt(diag(Lambda))./sigmas')
	CV_MAT_ELEM(*m, float, 0, 0) = sqrt(CV_MAT_ELEM(*eigenValues, float, 0, 0)) / CV_MAT_ELEM(*gaussKernel.sigma, float, 0, 0);	CV_MAT_ELEM(*m, float, 0, 1) = .0;
	CV_MAT_ELEM(*m, float, 1, 1) = sqrt(CV_MAT_ELEM(*eigenValues, float, 1, 0)) / CV_MAT_ELEM(*gaussKernel.sigma, float, 0, 1);	CV_MAT_ELEM(*m, float, 1, 0) = .0;

	CvMat* A = cvCreateMat(2, 2, CV_32FC1); // rotation + size with respect to the kernel
	cvMatMul(eigenVectors, m, A);

	// debug A
	/*CStdioFile f;	f.Open(L"A.txt", CFile::modeCreate | CFile::modeWrite);
	CString text;
	f.WriteString(L"covMat00    ");		text.Format(L"%4.4f   \n", CV_MAT_ELEM(*covMat, float, 0, 0));			f.WriteString(text);
	f.WriteString(L"covMat01    ");		text.Format(L"%4.4f   \n", CV_MAT_ELEM(*covMat, float, 0, 1));			f.WriteString(text);
	f.WriteString(L"covMat11    ");		text.Format(L"%4.4f   \n", CV_MAT_ELEM(*covMat, float, 1, 1));			f.WriteString(text);
	f.WriteString(L"\n");
	f.WriteString(L"kSigma00    ");		text.Format(L"%4.4f   \n", CV_MAT_ELEM(*gaussKernel.sigma, float, 0, 0));			f.WriteString(text);
	f.WriteString(L"kSigma01    ");		text.Format(L"%4.4f   \n", CV_MAT_ELEM(*gaussKernel.sigma, float, 0, 1));			f.WriteString(text);
	f.WriteString(L"\n");
	f.WriteString(L"eigVal00    ");		text.Format(L"%4.4f   \n", CV_MAT_ELEM(*eigenValues, float, 0, 0));		f.WriteString(text);
	f.WriteString(L"eigVal10    ");		text.Format(L"%4.4f   \n", CV_MAT_ELEM(*eigenValues, float, 1, 0));		f.WriteString(text);
	f.WriteString(L"\n");
	f.WriteString(L"eigVec00    ");		text.Format(L"%4.4f   \n", CV_MAT_ELEM(*eigenVectors, float, 0, 0));	f.WriteString(text);
	f.WriteString(L"eigVec01    ");		text.Format(L"%4.4f   \n", CV_MAT_ELEM(*eigenVectors, float, 0, 1));	f.WriteString(text);
	f.WriteString(L"eigVec10    ");		text.Format(L"%4.4f   \n", CV_MAT_ELEM(*eigenVectors, float, 1, 0));	f.WriteString(text);
	f.WriteString(L"eigVec11    ");		text.Format(L"%4.4f   \n", CV_MAT_ELEM(*eigenVectors, float, 1, 1));	f.WriteString(text);
	f.WriteString(L"\n");
	f.WriteString(L"m00        ");		text.Format(L"%4.4f   \n", CV_MAT_ELEM(*m, float, 0, 0));				f.WriteString(text);
	f.WriteString(L"m11        ");		text.Format(L"%4.4f   \n", CV_MAT_ELEM(*m, float, 1, 1));				f.WriteString(text);
	f.WriteString(L"\n");
	f.WriteString(L"A00        ");		text.Format(L"%4.4f   \n", CV_MAT_ELEM(*A, float, 0, 0));				f.WriteString(text);
	f.WriteString(L"A01        ");		text.Format(L"%4.4f   \n", CV_MAT_ELEM(*A, float, 0, 1));				f.WriteString(text);
	f.WriteString(L"A10        ");		text.Format(L"%4.4f   \n", CV_MAT_ELEM(*A, float, 1, 0));				f.WriteString(text);
	f.WriteString(L"A11        ");		text.Format(L"%4.4f   \n", CV_MAT_ELEM(*A, float, 1, 1));				f.WriteString(text);	
	f.Close();*/


	// -------- rotate and translate --------
	CvMat* rXrY = cvCreateMat(2, kernelWidth, CV_32FC1);
	for (int i = 0; i < kernelWidth; i++) {
		CV_MAT_ELEM(*rXrY, float, 0, i) = float(CV_MAT_ELEM(*gaussKernel.rX, int, 0, i));
		CV_MAT_ELEM(*rXrY, float, 1, i) = float(CV_MAT_ELEM(*gaussKernel.rY, int, 0, i));
	}

	CvMat* Z = cvCreateMat(2, kernelWidth, CV_32FC1); // rotation matrix
	cvMatMul(A, rXrY, Z);

	for (int i = 0; i < kernelWidth; i++) {
		CV_MAT_ELEM(*newRx, float, 0, i) = CV_MAT_ELEM(*Z, float, 0, i) + float(mean.x);
		CV_MAT_ELEM(*newRy, float, 0, i) = CV_MAT_ELEM(*Z, float, 1, i) + float(mean.y);
	}

	
	// -------- interpolate(?) --------
	//interp2FromMatlabEngine(srcImg, newRx, newRy, imgROIchannelR, imgROIchannelG, imgROIchannelB);
	interp2(srcImg, newRx, newRy, imgROIchannelR, imgROIchannelG, imgROIchannelB);

	// debug imgROI
	CStdioFile f;	f.Open(L"imgROI.txt", CFile::modeCreate | CFile::modeWrite);	CString text;
	for (int i = 0; i < kernelWidth; i++) { 
		text.Format(L"%4.4f   ", CV_MAT_ELEM(*imgROIchannelR, float, 0, i));		f.WriteString(text);
		text.Format(L"%4.4f   ", CV_MAT_ELEM(*imgROIchannelG, float, 0, i));		f.WriteString(text);
		text.Format(L"%4.4f \n", CV_MAT_ELEM(*imgROIchannelB, float, 0, i));		f.WriteString(text);
	} f.Close();


	// -------- newRx, newRy --------
	for (int i = 0; i < kernelWidth; i++) {
		CV_MAT_ELEM(*newRx, float, 0, i) = CV_MAT_ELEM(*Z, float, 0, i);
		CV_MAT_ELEM(*newRy, float, 0, i) = CV_MAT_ELEM(*Z, float, 1, i);
	}

	// debug newRx
	/*CStdioFile f2;	f2.Open(L"newRy.txt", CFile::modeCreate | CFile::modeWrite);	CString text2;
	for (int i = 0; i < kernelWidth; i++) { 
		text2.Format(L"%4.4f   ", CV_MAT_ELEM(*newRy, float, 0, i));		f2.WriteString(text2);		
	} f2.Close();*/


	// -------- release --------	
	cvReleaseMat(&eigenValues);
	cvReleaseMat(&eigenVectors);
	cvReleaseMat(&m);
	cvReleaseMat(&A);
	cvReleaseMat(&rXrY);
	cvReleaseMat(&Z);
}

void MeanShiftEMScaleKalman::getHistogram(CvMat* imgROIchannelR, CvMat* imgROIchannelG, CvMat* imgROIchannelB, Histogram *histogram) {
	int sqrN_BINS = N_BINS*N_BINS;
	int kernelWidth = gaussKernel.rK->width;


	// ------ init histogram ------
	int numOfBins = sqrN_BINS*N_BINS;
	histogram->data = cvCreateMat(1, numOfBins, CV_32FC1);	
	histogram->originH = cvCreateMat(kernelWidth, numOfBins, CV_32FC1);
	for (int i = 0; i < kernelWidth; i++) for (int j = 0; j < numOfBins; j++) CV_MAT_ELEM(*histogram->originH, float, i, j) = 0;


	// ------ make histogram ------
	double nDiv = 256.0 / N_BINS;	
	for (int i = 0; i < kernelWidth; i++) {
		int iBin = sqrN_BINS * cvFloor(CV_MAT_ELEM(*imgROIchannelR, float, 0, i) / nDiv) + 
					  N_BINS * cvFloor(CV_MAT_ELEM(*imgROIchannelG, float, 0, i) / nDiv) + 
					 		   cvFloor(CV_MAT_ELEM(*imgROIchannelB, float, 0, i) / nDiv);
		CV_MAT_ELEM(*histogram->originH, float, i, iBin) = 1;
	}

	// debug histogram->originH
	/*CStdioFile f0;	f0.Open(L"histogram_originH.txt", CFile::modeCreate | CFile::modeWrite); CString text0;
	for (int i = 0; i < kernelWidth; i++) {
		for (int j = 0; j < numOfBins; j++) {
			text0.Format(L"%4.4f   ", CV_MAT_ELEM(*histogram->originH, float, i, j));	f0.WriteString(text0);
		} f0.WriteString(L"\n");
	} f0.Close();*/

	cvMatMul(gaussKernel.rK, histogram->originH, histogram->data);

	// debug histogram->data
	/*CStdioFile f;	f.Open(L"histogram_data.txt", CFile::modeCreate | CFile::modeWrite); CString text;
	for (int i = 0; i < numOfBins; i++) {
		text.Format(L"%4.4f   ", CV_MAT_ELEM(*histogram->data, float, 0, i));	f.WriteString(text);
	} f.Close();*/
}

void MeanShiftEMScaleKalman::convertMeanAndVarianceToStateVector(CvPoint mean, CvMat *covMat, CvMat* stateVector) { // stateVector = [ m1 m2 cov11 cov12 cov22 ]
	CV_MAT_ELEM(*stateVector, float, 0, 0) = float(mean.x);
	CV_MAT_ELEM(*stateVector, float, 0, 1) = float(mean.y);

	CvMat* choleskyMat = cvCreateMat(2, 2, CV_32FC1);
	choleskyDecompose(covMat, choleskyMat);
	CV_MAT_ELEM(*stateVector, float, 0, 2) = CV_MAT_ELEM(*choleskyMat, float, 0, 0);
	CV_MAT_ELEM(*stateVector, float, 0, 3) = CV_MAT_ELEM(*choleskyMat, float, 0, 1);
	CV_MAT_ELEM(*stateVector, float, 0, 4) = CV_MAT_ELEM(*choleskyMat, float, 1, 1);

	cvReleaseMat(&choleskyMat);

	// debug stateVector
	/*float covMat00 = CV_MAT_ELEM(*covMat, float, 0, 0);
	float covMat01 = CV_MAT_ELEM(*covMat, float, 0, 1);
	float covMat11 = CV_MAT_ELEM(*covMat, float, 1, 1);
	float s2 = CV_MAT_ELEM(*stateVector, float, 0, 2);
	float s3 = CV_MAT_ELEM(*stateVector, float, 0, 3);
	float s4 = CV_MAT_ELEM(*stateVector, float, 0, 4);*/
}

void MeanShiftEMScaleKalman::convertStateVectorToMeanAndVariance(CvMat* stateVector, CvPoint* mean, CvMat *covMat) {
	mean->x = (int)CV_MAT_ELEM(*stateVector, float, 0, 0);
	mean->y = (int)CV_MAT_ELEM(*stateVector, float, 0, 1);

	CV_MAT_ELEM(*covMat, float, 0, 0) = sqr(CV_MAT_ELEM(*stateVector, float, 0, 2));
	CV_MAT_ELEM(*covMat, float, 0, 1) = CV_MAT_ELEM(*stateVector, float, 0, 2) * CV_MAT_ELEM(*stateVector, float, 0, 3);
	CV_MAT_ELEM(*covMat, float, 1, 0) = CV_MAT_ELEM(*covMat, float, 0, 1);
	CV_MAT_ELEM(*covMat, float, 1, 1) = sqr(CV_MAT_ELEM(*stateVector, float, 0, 3)) + sqr(CV_MAT_ELEM(*stateVector, float, 0, 4));

	// debug covMat
	/*float covMat00 = CV_MAT_ELEM(*covMat, float, 0, 0);
	float covMat01 = CV_MAT_ELEM(*covMat, float, 0, 1);
	float covMat11 = CV_MAT_ELEM(*covMat, float, 1, 1);
	float s2 = CV_MAT_ELEM(*stateVector, float, 0, 2);
	float s3 = CV_MAT_ELEM(*stateVector, float, 0, 3);
	float s4 = CV_MAT_ELEM(*stateVector, float, 0, 4);*/
}	

void MeanShiftEMScaleKalman::EMShift(IplImage* srcImg, CvPoint mean, CvMat* covMat, 
									 CvPoint* newMean, CvMat *newCovMat) {
	// Calculate gradient step to maximize similarity between the target and the object histogram - Bhattacharayya coefficient. The step is in both position  
	// and scale as described in (original mean shift does only position - so if you do not use the resulting covMat, you will get the standard mean shift)

	int kernelWidth = gaussKernel.rX->width;


	// ------ get target region ------
	// "empty" imgROI
	CvMat* imgROIchannelR = cvCreateMat(1, kernelWidth, CV_32FC1); // gaussKernel.rX->height = 1
	CvMat* imgROIchannelG = cvCreateMat(1, kernelWidth, CV_32FC1);
	CvMat* imgROIchannelB = cvCreateMat(1, kernelWidth, CV_32FC1);

	// newRX, newRY
	CvMat* newRX = cvCreateMat(1, kernelWidth, CV_32FC1);
	CvMat* newRY = cvCreateMat(1, kernelWidth, CV_32FC1);

	getAffineRegion(srcImg, mean, covMat, imgROIchannelR, imgROIchannelG, imgROIchannelB, newRX, newRY);

	// debug newRx, newRy
	/*CStdioFile f0;	f0.Open(L"newRx.txt", CFile::modeCreate | CFile::modeWrite);	CString text0;
	CStdioFile f1;	f1.Open(L"newRy.txt", CFile::modeCreate | CFile::modeWrite);	CString text1;
	for (int i = 0; i < kernelWidth; i++) { 
		text0.Format(L"%4.4f   ", CV_MAT_ELEM(*newRX, float, 0, i));		f0.WriteString(text0);
		text1.Format(L"%4.4f   ", CV_MAT_ELEM(*newRY, float, 0, i));		f1.WriteString(text1);
	} f0.Close(); f1.Close();*/


	// ------ get histogram and mapping H pixels to bins ------	
	Histogram regionHistogram;		getHistogram(imgROIchannelR, imgROIchannelG, imgROIchannelB, &regionHistogram);
	int histWidth = regionHistogram.data->width;

	// debug regionHistogram.originH
	/*CStdioFile f0;	f0.Open(L"regionHistogram.H.txt", CFile::modeCreate | CFile::modeWrite);	CString text0;
	for (int i = 0; i < newRX->width; i++) {
		for (int j = 0; j < N_BINS*N_BINS*N_BINS; j++) {
			text0.Format(L"%4.4f   ", CV_MAT_ELEM(*regionHistogram.originH, float, i, j));	f0.WriteString(text0);
		} f0.WriteString(L"\n");
	} f0.Close();*/

	// debug histogram->data
	/*CStdioFile fr;	fr.Open(L"regHistogram_data.txt", CFile::modeCreate | CFile::modeWrite); CString textr;
	CStdioFile fo;	fo.Open(L"objHistogram_data.txt", CFile::modeCreate | CFile::modeWrite); CString texto;
	for (int i = 0; i < histWidth; i++) {
		textr.Format(L"%4.4f   ", CV_MAT_ELEM(*regionHistogram.data, float, 0, i));	fr.WriteString(textr);
		texto.Format(L"%4.4f   ", CV_MAT_ELEM(*objHistogram.data, float, 0, i));	fo.WriteString(texto);
	} fr.Close(); fo.Close();*/


	// ------ A = sqrt(histO.data(riNonZero)./histR.data(riNonZero)) ------
	CvMat* sqrtHistOAndHistR = cvCreateMat(1, histWidth, CV_32FC1);
	for (int i = 0; i < histWidth; i++) {
		float regHistogramValue = CV_MAT_ELEM(*regionHistogram.data, float, 0, i);
		if (regHistogramValue > 0) {
			float objHistogramValue = CV_MAT_ELEM(*objHistogram.data, float, 0, i);
			CV_MAT_ELEM(*sqrtHistOAndHistR, float, 0, i) = sqrt(objHistogramValue / regHistogramValue);
		}
		else CV_MAT_ELEM(*sqrtHistOAndHistR, float, 0, i) = 0;
	}
	

	// ------ A' ------
	CvMat* sqrtHistOAndHistRTranspose = cvCreateMat(histWidth, 1, CV_32FC1);
	cvTranspose(sqrtHistOAndHistR, sqrtHistOAndHistRTranspose);

	// debug sqrtHistOAndHistRTranspose
	CStdioFile f1;	f1.Open(L"sqrtHistOAndHistRTranspose.txt", CFile::modeCreate | CFile::modeWrite); CString text1;
	for (int i = 0; i < histWidth; i++) {
		text1.Format(L"%4.4f   ", CV_MAT_ELEM(*sqrtHistOAndHistRTranspose, float, i, 0));	f1.WriteString(text1);
	} f1.Close();


	// ------ histR.H(:,riNonZero) * A' ------
	CvMat* weights = cvCreateMat(kernelWidth, 1, CV_32FC1);
	cvMatMul(regionHistogram.originH, sqrtHistOAndHistRTranspose, weights);

	// debug weights
	CStdioFile f;	f.Open(L"weights.txt", CFile::modeCreate | CFile::modeWrite);	CString text;
	for (int i = 0; i < newRX->width; i++) {
		text.Format(L"%4.4f   ", CV_MAT_ELEM(*weights, float, i, 0));	f.WriteString(text);
	} f.Close(); 


	// ------ rWeigths = (histR.H(:,riNonZero)*sqrt(histO.data(riNonZero)./histR.data(riNonZero))')'.*K.rK ------
	CvMat* wg = cvCreateMat(1, kernelWidth, CV_32FC1);
	float sumWg = 0;
	for (int i = 0; i < newRX->width; i++) {
		//float a = CV_MAT_ELEM(*gaussKernel.rK, float, 0, i);
		//float b = CV_MAT_ELEM(*weights, float, i, 0);
		//float c = a*b;
		CV_MAT_ELEM(*wg, float, 0, i) = CV_MAT_ELEM(*weights, float, i, 0) * CV_MAT_ELEM(*gaussKernel.rK, float, 0, i);
		sumWg += CV_MAT_ELEM(*wg, float, 0, i);
	}

	// debug weights
	/*CStdioFile f;	f.Open(L"wg.txt", CFile::modeCreate | CFile::modeWrite);	CString text;
	for (int i = 0; i < newRX->width; i++) {
		text.Format(L"%4.4f   ", CV_MAT_ELEM(*wg, float, 0, i));	f.WriteString(text);
	} f.Close(); */


	// ------ q ------
	CvMat* q = cvCreateMat(1, kernelWidth, CV_32FC1);
	if (sumWg > 0) 
		for (int i = 0; i < kernelWidth; i++) CV_MAT_ELEM(*q, float, 0, i) = CV_MAT_ELEM(*wg, float, 0, i) / sumWg;
	else
		for (int i = 0; i < kernelWidth; i++) CV_MAT_ELEM(*q, float, 0, i) = CV_MAT_ELEM(*wg, float, 0, i);	

	// debug q
	/*CStdioFile f;	f.Open(L"q.txt", CFile::modeCreate | CFile::modeWrite);	CString text;
	for (int i = 0; i < kernelWidth; i++) {
		text.Format(L"%4.4f   ", CV_MAT_ELEM(*q, float, 0, i));	f.WriteString(text);
	} f.Close(); */


	// ------ dx = [sum(rQs.*rX);sum(rQs.*rY)] ------
	float dx = 0, dy = 0;
	for (int i = 0; i < kernelWidth; i++) {
		float aQ = CV_MAT_ELEM(*q, float, 0, i);
		dx += aQ * CV_MAT_ELEM(*newRX, float, 0, i);
		dy += aQ * CV_MAT_ELEM(*newRY, float, 0, i);
	}
	newMean->x += int(dx);
	newMean->y += int(dy);


	// ------ covMat V ------
	/*float vxx = 0, vyy = 0, vxy = 0;
	for (int i = 0; i < kernelWidth; i++) {
		float eachQ = CV_MAT_ELEM(*q, float, 0, i);
		float eachRX = CV_MAT_ELEM(*newRX, float, 0, i);
		float eachRY = CV_MAT_ELEM(*newRY, float, 0, i);		
		vxx += eachQ + 2*eachRX;
		vyy += eachQ + 2*eachRY;
		vxy += eachQ + eachRX + eachRY;
	}
	
	CV_MAT_ELEM(*newCovMat, float, 0, 0) = float(max(BETA*vxx, 1.0)); // clip to min 1
	CV_MAT_ELEM(*newCovMat, float, 1, 1) = float(max(BETA*vyy, 1.0));
	CV_MAT_ELEM(*newCovMat, float, 0, 1) = CV_MAT_ELEM(*newCovMat, float, 1, 0) = vxy;*/

	// giu nguyen covMat
	CV_MAT_ELEM(*newCovMat, float, 0, 0) = CV_MAT_ELEM(*covMat, float, 0, 0);
	CV_MAT_ELEM(*newCovMat, float, 0, 1) = CV_MAT_ELEM(*covMat, float, 0, 1);
	CV_MAT_ELEM(*newCovMat, float, 1, 0) = CV_MAT_ELEM(*covMat, float, 1, 0);
	CV_MAT_ELEM(*newCovMat, float, 1, 1) = CV_MAT_ELEM(*covMat, float, 1, 1);

	// debug covMat V
	/*CStdioFile f;	f.Open(L"v.txt", CFile::modeCreate | CFile::modeWrite);		CString text;
	text.Format(L"%4.4f   ", CV_MAT_ELEM(*newCovMat, float, 0, 0));				f.WriteString(text);
	text.Format(L"%4.4f \n", CV_MAT_ELEM(*newCovMat, float, 0, 1));				f.WriteString(text);
	text.Format(L"%4.4f   ", CV_MAT_ELEM(*newCovMat, float, 1, 0));				f.WriteString(text);
	text.Format(L"%4.4f", CV_MAT_ELEM(*newCovMat, float, 1, 1));				f.WriteString(text);
	f.Close(); */


	// ==== release ====
	cvReleaseMat(&imgROIchannelR);
	cvReleaseMat(&imgROIchannelG);
	cvReleaseMat(&imgROIchannelB);
	cvReleaseMat(&newRX);
	cvReleaseMat(&newRY);
	cvReleaseMat(&sqrtHistOAndHistR);
	cvReleaseMat(&sqrtHistOAndHistRTranspose);
	cvReleaseMat(&weights);
	cvReleaseMat(&q);
}

void MeanShiftEMScaleKalman::fitQuadraticToBhattacharyaCoeff(IplImage* srcImg, CvMat* stateVector, CvMat* R) { // tra ve ma tran duong cheo 5x5 
	float sqrLambda = sqr(float(LAMBDA));

	//
	CvPoint mean = cvPoint(0, 0);
	CvMat* covMat = cvCreateMat(2, 2, CV_32FC1); 
	convertStateVectorToMeanAndVariance(stateVector, &mean, covMat);
	float rho = calculateBhattacharayyaCoeff(srcImg, mean, covMat);

	
	// --------- position (mean) ---------
	const int d = 2, sqrD = 4;

	// -- x --
	CV_MAT_ELEM(*stateVector, float, 0, 0) = CV_MAT_ELEM(*stateVector, float, 0, 0) + d;	convertStateVectorToMeanAndVariance(stateVector, &mean, covMat);
	float sample1 = rho - calculateBhattacharayyaCoeff(srcImg, mean, covMat);				if (sample1 < FCLIP) sample1 = float(FCLIP); // clip negative(?)

	CV_MAT_ELEM(*stateVector, float, 0, 0) = CV_MAT_ELEM(*stateVector, float, 0, 0) - 2*d;	convertStateVectorToMeanAndVariance(stateVector, &mean, covMat);
	float sample2 = rho - calculateBhattacharayyaCoeff(srcImg, mean, covMat);				if (sample2 < FCLIP) sample2 = float(FCLIP); // clip negative(?)

	CV_MAT_ELEM(*R, float, 0, 0) = float( (sqrD/sample1 + sqrD/sample2) / 2.0 * sqrLambda ); // * sqr(LAMBDA);

	// -- y --
	CV_MAT_ELEM(*stateVector, float, 0, 1) = CV_MAT_ELEM(*stateVector, float, 0, 1) + d;	convertStateVectorToMeanAndVariance(stateVector, &mean, covMat);
	sample1 = rho - calculateBhattacharayyaCoeff(srcImg, mean, covMat);						if (sample1 < FCLIP) sample1 = float(FCLIP); // clip negative(?)

	CV_MAT_ELEM(*stateVector, float, 0, 1) = CV_MAT_ELEM(*stateVector, float, 0, 1) - 2*d;	convertStateVectorToMeanAndVariance(stateVector, &mean, covMat);
	sample2 = rho - calculateBhattacharayyaCoeff(srcImg, mean, covMat);						if (sample2 < FCLIP) sample2 = float(FCLIP); // clip negative(?)

	CV_MAT_ELEM(*R, float, 0, 1) = float( (sqrD/sample1 + sqrD/sample2) / 2.0 * sqrLambda ); // * sqr(LAMBDA);


	// --------- variance ---------
	float zeroDotOneFloat = float(0.1);

	// -- vxx --
	float df = zeroDotOneFloat * CV_MAT_ELEM(*stateVector, float, 0, 2);
	float sqrDf = sqr(df);

	CV_MAT_ELEM(*stateVector, float, 0, 2) = CV_MAT_ELEM(*stateVector, float, 0, 2) + df;	convertStateVectorToMeanAndVariance(stateVector, &mean, covMat);
	sample1 = rho - calculateBhattacharayyaCoeff(srcImg, mean, covMat);						if (sample1 < FCLIP) sample1 = float(FCLIP); // clip negative(?)

	CV_MAT_ELEM(*stateVector, float, 0, 2) = CV_MAT_ELEM(*stateVector, float, 0, 2) - 2*df;	convertStateVectorToMeanAndVariance(stateVector, &mean, covMat);
	sample2 = rho - calculateBhattacharayyaCoeff(srcImg, mean, covMat);						if (sample2 < FCLIP) sample2 = float(FCLIP); // clip negative(?)

	CV_MAT_ELEM(*R, float, 0, 2) = float( (sqrDf/sample1 + sqrDf/sample2) / 2.0 * sqrLambda ); // * sqr(LAMBDA);

	// -- vxy --
	df = zeroDotOneFloat * CV_MAT_ELEM(*stateVector, float, 0, 3);
	sqrDf = sqr(df);

	CV_MAT_ELEM(*stateVector, float, 0, 3) = CV_MAT_ELEM(*stateVector, float, 0, 3) + df;	convertStateVectorToMeanAndVariance(stateVector, &mean, covMat);
	sample1 = rho - calculateBhattacharayyaCoeff(srcImg, mean, covMat);						if (sample1 < FCLIP) sample1 = float(FCLIP); // clip negative(?)

	CV_MAT_ELEM(*stateVector, float, 0, 3) = CV_MAT_ELEM(*stateVector, float, 0, 3) - 2*df;	convertStateVectorToMeanAndVariance(stateVector, &mean, covMat);
	sample2 = rho - calculateBhattacharayyaCoeff(srcImg, mean, covMat);						if (sample2 < FCLIP) sample2 = float(FCLIP); // clip negative(?)

	CV_MAT_ELEM(*R, float, 0, 3) = float( (sqrDf/sample1 + sqrDf/sample2) / 2.0 * sqrLambda ); // * sqr(LAMBDA);

	// -- vyy --
	df = zeroDotOneFloat * CV_MAT_ELEM(*stateVector, float, 0, 4);
	sqrDf = sqr(df);

	CV_MAT_ELEM(*stateVector, float, 0, 4) = CV_MAT_ELEM(*stateVector, float, 0, 4) + df;	convertStateVectorToMeanAndVariance(stateVector, &mean, covMat);
	sample1 = rho - calculateBhattacharayyaCoeff(srcImg, mean, covMat);						if (sample1 < FCLIP) sample1 = float(FCLIP); // clip negative(?)

	CV_MAT_ELEM(*stateVector, float, 0, 4) = CV_MAT_ELEM(*stateVector, float, 0, 4) - 2*df;	convertStateVectorToMeanAndVariance(stateVector, &mean, covMat);
	sample2 = rho - calculateBhattacharayyaCoeff(srcImg, mean, covMat);						if (sample2 < FCLIP) sample2 = float(FCLIP); // clip negative(?)

	CV_MAT_ELEM(*R, float, 0, 4) = float( (sqrDf/sample1 + sqrDf/sample2) / 2.0 * sqrLambda ); // * sqr(LAMBDA);


	// ==== release ====
	cvReleaseMat(&covMat);
}

float MeanShiftEMScaleKalman::calculateBhattacharayyaCoeff(IplImage* srcImg, CvPoint mean, CvMat* covMat) {

	int kernelWidth = gaussKernel.rX->width;

	// ------ get target region ------
	// "empty" imgROI
	CvMat* imgROIchannelR = cvCreateMat(1, kernelWidth, CV_32FC1); // gaussKernel.rX->height = 1
	CvMat* imgROIchannelG = cvCreateMat(1, kernelWidth, CV_32FC1);
	CvMat* imgROIchannelB = cvCreateMat(1, kernelWidth, CV_32FC1);

	// newRX, newRY
	CvMat* newRX = cvCreateMat(1, kernelWidth, CV_32FC1); // NOT USED
	CvMat* newRY = cvCreateMat(1, kernelWidth, CV_32FC1); // NOT USED

	getAffineRegion(srcImg, mean, covMat, imgROIchannelR, imgROIchannelG, imgROIchannelB, newRX, newRY);

	// ------ get histogram and mapping H pixels to bins ------	
	Histogram regionHistogram;
	getHistogram(imgROIchannelR, imgROIchannelG, imgROIchannelB, &regionHistogram);

	// ======= calculate rho =======
	float rho = 0;
	for (int i = 0; i < regionHistogram.data->width; i++) {
		float binOfObjHist = CV_MAT_ELEM(*objHistogram.data, float, 0, i);
		float binOfRegionHist = CV_MAT_ELEM(*regionHistogram.data, float, 0, i);
		rho += sqrt(binOfObjHist * binOfRegionHist);
	}

	// ==== release ====
	cvReleaseMat(&imgROIchannelR);
	cvReleaseMat(&imgROIchannelG);
	cvReleaseMat(&imgROIchannelB);
	cvReleaseMat(&newRX);
	cvReleaseMat(&newRY);

	return rho;
}

void MeanShiftEMScaleKalman::drawTracker(IplImage* srcImg, CvMat* stateVector) {
	double x = double(CV_MAT_ELEM(*stateVector, float, 0, 0));
	double y = double(CV_MAT_ELEM(*stateVector, float, 0, 1));
	double vxx = double(CV_MAT_ELEM(*stateVector, float, 0, 2));
	double vxy = double(CV_MAT_ELEM(*stateVector, float, 0, 3));
	double vyy = double(CV_MAT_ELEM(*stateVector, float, 0, 4));

	CvBox2D box;
	box.center = cvPoint2D32f(x, y);
	box.size = cvSize2D32f(vxx*2+1, vyy*2+1);	
	box.angle = 90;
	cvEllipseBox(srcImg, box, CV_RGB(0,255,0), 1);
}

void MeanShiftEMScaleKalman::choleskyDecompose(CvMat* covMat, CvMat* choleskyMat) { // chi dung cho ma tran 2x2 va postitive definite
	// ref: Numerical Recipes in C, 2nd edition, 1992

	int cols = covMat->cols;
	float* data = covMat->data.fl;

	float sum = 0;
	int k;
	for (int i = 0; i < cols; i++) 
		for (int j = i; j < cols; j++) {
			for (sum = data[i*cols + j], k = i - 1; k >= 0; k--) 
				sum -= data[i*cols + k] * data[j*cols + k]; // if sum <= 0, covMat is not postitive definite

			if (i == j) CV_MAT_ELEM(*choleskyMat, float, i, j) = sqrt(sum);
			else {
				CV_MAT_ELEM(*choleskyMat, float, i, j) = sum/CV_MAT_ELEM(*choleskyMat, float, i, i);
				CV_MAT_ELEM(*choleskyMat, float, j, i) = 0; // ntt
			}
		}
}

void MeanShiftEMScaleKalman::generateArraysFor3DPlot(int sizX, int sizY, CvMat* xArray, CvMat* yArray) { // ~ meshgrid (Matlab)
	for (int i = -sizY; i <= sizY; i++) 
		for (int j = -sizX; j <= sizX; j++) {
			int iArray = i + sizY;
			int jArray = j + sizX;
			CV_MAT_ELEM(*xArray, int, iArray, jArray) = i;
			CV_MAT_ELEM(*yArray, int, iArray, jArray) = j;
		}
}

void MeanShiftEMScaleKalman::reshapeKernelToOneRowFloat(CvMat* src, CvMat* dst) { // src: n x m; dst: 1 x m*n
	int dstIndex = 0;
	for (int j = 0; j < src->width; j++) 
		for (int i = 0; i < src->height; i++) {
			CV_MAT_ELEM(*dst, float, 0, dstIndex) = CV_MAT_ELEM(*src, float, i, j);
			dstIndex++;
		}
}

void MeanShiftEMScaleKalman::reshapeKernelToOneRowInt(CvMat* src, CvMat* dst) { // src: n x m; dst: 1 x m*n
	int dstIndex = 0;
	for (int j = 0; j < src->width; j++) 
		for (int i = 0; i < src->height; i++) {
			CV_MAT_ELEM(*dst, int, 0, dstIndex) = CV_MAT_ELEM(*src, int, i, j);
			dstIndex++;
		}
}

void MeanShiftEMScaleKalman::divideMatrixByANumber(CvMat* src, CvMat* dst, float n) {
	for (int i = 0; i < src->height; i++) 
		for (int j = 0; j < src->width; j++) 
			CV_MAT_ELEM(*dst, float, i, j) = float(CV_MAT_ELEM(*src, int, i, j)) / n; // <--- int chuyen thanh float
}

void MeanShiftEMScaleKalman::generateIdentityMatrix(CvMat* mat) {
	for (int i = 0; i < mat->height; i++) {
		for (int j = 0; j < mat->width; j++) {
			if (i == j) CV_MAT_ELEM(*mat, float, i, j) = 1;
			else CV_MAT_ELEM(*mat, float, i, j) = 0;
		}
	}
}

float MeanShiftEMScaleKalman::sqr(float n) {
	return n*n;
}

void MeanShiftEMScaleKalman::generateCircleWithSigmaRadius(CvMat *rCircle) {
	int nPoints = rCircle->width;

	CV_MAT_ELEM(*rCircle, float, 0, 0) = SIGMA; // = SIGMA*cos(0)
	CV_MAT_ELEM(*rCircle, float, 1, 0) = 0;		// = SIGMA*sin(0)

	float dur = float(2*CV_PI)/float(nPoints-1);
	for (int i = 1; i < nPoints; i++) {
		CV_MAT_ELEM(*rCircle, float, 0, i) = float(SIGMA*cos(dur*i)); // neu khong nhan voi SIGMA thi vong tron tao thanh la vong tron don vi (ban kinh = 1)
		CV_MAT_ELEM(*rCircle, float, 1, i) = float(SIGMA*sin(dur*i));
	}

	// debug rCircle
	/*CStdioFile f;	f.Open(L"rCircle.txt", CFile::modeCreate | CFile::modeWrite); CString text;
	for (int i = 0; i < nPoints; i++) { 
		text.Format(L"%4.4f   ", CV_MAT_ELEM(*rCircle, float, 0, i));		f.WriteString(text);
		text.Format(L"%4.4f \n", CV_MAT_ELEM(*rCircle, float, 1, i));		f.WriteString(text);
	} f.Close();*/
}

void MeanShiftEMScaleKalman::interp2(IplImage* srcImg, CvMat* rX, CvMat* rY, CvMat* imgROIr, CvMat* imgROIg, CvMat* imgROIb) {		
	uchar* imgDataChar = (uchar *)srcImg->imageData;
	int imgStep = srcImg->widthStep;
	int imgNChannels = srcImg->nChannels;
	int imgSize = srcImg->height * srcImg->width;

	for (int i = 0; i < rX->width; i++) { // bilinear interpolation
		float eachRX = CV_MAT_ELEM(*rX, float, 0, i);		int floorEachRX = cvFloor(eachRX);	int ceilEachRX = cvCeil(eachRX);
		float eachRY = CV_MAT_ELEM(*rY, float, 0, i);		int floorEachRY = cvFloor(eachRY);	int ceilEachRY = cvCeil(eachRY);

		CV_MAT_ELEM(*imgROIr, float, 0, i) = float(imgDataChar[floorEachRX*imgStep + floorEachRY*imgNChannels] 
												 + imgDataChar[floorEachRX*imgStep + ceilEachRY*imgNChannels]
												 + imgDataChar[ceilEachRX*imgStep + floorEachRY*imgNChannels] 
												 + imgDataChar[ceilEachRX*imgStep + ceilEachRY*imgNChannels]) / 4;
		CV_MAT_ELEM(*imgROIg, float, 0, i) = float(imgDataChar[floorEachRX*imgStep + floorEachRY*imgNChannels + 1] 
												 + imgDataChar[floorEachRX*imgStep + ceilEachRY*imgNChannels + 1]
												 + imgDataChar[ceilEachRX*imgStep + floorEachRY*imgNChannels + 1] 
												 + imgDataChar[ceilEachRX*imgStep + ceilEachRY*imgNChannels + 1]) / 4;
		CV_MAT_ELEM(*imgROIb, float, 0, i) = float(imgDataChar[floorEachRX*imgStep + floorEachRY*imgNChannels + 2] 
												 + imgDataChar[floorEachRX*imgStep + ceilEachRY*imgNChannels + 2]
												 + imgDataChar[ceilEachRX*imgStep + floorEachRY*imgNChannels + 2] 
												 + imgDataChar[ceilEachRX*imgStep + ceilEachRY*imgNChannels + 2]) / 4;		
	}
}

void MeanShiftEMScaleKalman::interp2FromMatlabCMathLibrary(IplImage* srcImg, CvMat* rX, CvMat* rY, CvMat* imgROIr, CvMat* imgROIg, CvMat* imgROIb) {
/*	// ------------- prepare the input -------------
	char met[] = "linear";

	// prepare srcImg, tao 3 mang R,G,B khac nhau
	uchar* imgDataChar = (uchar *)srcImg->imageData;
	int imgStep = srcImg->widthStep;
	int imgNChannels = srcImg->nChannels;
	int imgSize = srcImg->height * srcImg->width;
	
	double* imgDataR = new double[imgSize];
	double* imgDataG = new double[imgSize];
	double* imgDataB = new double[imgSize];

	int imgSizeIndex = 0;	
	for (int i = 0; i < srcImg->height; i++) {
		for (int j = 0; j < srcImg->width; j++) {
			imgDataR[imgSizeIndex] = imgDataChar[i*imgStep + j*imgNChannels];
			imgDataG[imgSizeIndex] = imgDataChar[i*imgStep + j*imgNChannels + 1];
			imgDataB[imgSizeIndex] = imgDataChar[i*imgStep + j*imgNChannels + 2];
			imgSizeIndex++;
		}
	}

	// prepare rX, rY
	float *rXDataFloat = rX->data.fl;	double *rXData = new double[rX->width];
	float *rYDataFloat = rY->data.fl;	double *rYData = new double[rX->width];
	for (int i = 0; i < rX->width; i++) { 
		rXData[i] = rXDataFloat[i]; // double = float
		rYData[i] = rYDataFloat[i];
	}

	// ------------- nap vao ham matlab -------------
	mxArray *method;
	mxArray *Z1, *Z2, *Z3, *X, *Y;
	mxArray *ZI1, *ZI2, *ZI3; 

	mlfAssign(&method, mxCreateString(met));
	mlfAssign(&Z1, mlfDoubleMatrix(srcImg->height, srcImg->width, imgDataR, NULL));
	mlfAssign(&Z2, mlfDoubleMatrix(srcImg->height, srcImg->width, imgDataG, NULL));
	mlfAssign(&Z3, mlfDoubleMatrix(srcImg->height, srcImg->width, imgDataB, NULL));
	mlfAssign(&X, mlfDoubleMatrix(1, rX->width, rXData, NULL));
	mlfAssign(&Y, mlfDoubleMatrix(1, rX->width, rYData, NULL));
	
	//mlfAssign(&ZI1, mlfInterp2(Z1, X, Y, method));
	//mlfAssign(&ZI2, mlfInterp2(Z2, X, Y, method));
	//mlfAssign(&ZI3, mlfInterp2(Z3, X, Y, method));

	//mlfEps();
	mlfInterp2(Z1, 2, NULL, NULL, NULL, NULL); // ko chay duoc
	

	// ------------- tra ve imgROI -------------
	double *tmpR, *tmpG, *tmpB;
	tmpR = mxGetPr(ZI1);
	tmpG = mxGetPr(ZI2);
	tmpB = mxGetPr(ZI3);
	for (int i = 0; i < rX->width; i++) {
		CV_MAT_ELEM(*imgROIr, float, 0, i) = float(tmpR[i]);
		CV_MAT_ELEM(*imgROIg, float, 0, i) = float(tmpG[i]);
		CV_MAT_ELEM(*imgROIb, float, 0, i) = float(tmpB[i]);
	}

	// ------------- release -------------
	mxDestroyArray(Z1);
	mxDestroyArray(Z2);
	mxDestroyArray(Z3);
	mxDestroyArray(X);
	mxDestroyArray(Y);
	mxDestroyArray(ZI1);
	mxDestroyArray(ZI2);
	mxDestroyArray(ZI3);
	mxDestroyArray(method);*/
}

void MeanShiftEMScaleKalman::interp2FromMatlabEngine(IplImage* srcImg, CvMat* rX, CvMat* rY, CvMat* imgROIr, CvMat* imgROIg, CvMat* imgROIb) {
/*	int channels = srcImg->nChannels;	int step = srcImg->widthStep;	int width = srcImg->width;	int height = srcImg->height;	int imgSize = width*height;
	const uchar* imgData = (uchar *)srcImg->imageData;
	
	// ------- tao 3 array dai dien cho 3 channels ------

	// debug channel
	//CStdioFile f;	f.Open(L"channel.txt", CFile::modeCreate | CFile::modeWrite); CString text;

	double* channelR = new double[imgSize];		double* channelG = new double[imgSize];		double* channelB = new double[imgSize];
	int index = 0;
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++) {
			channelR[index] = imgData[i*step + j*channels] * 1.0;
			channelG[index] = imgData[i*step + j*channels + 1] * 1.0;
			channelB[index] = imgData[i*step + j*channels + 2] * 1.0;
			//text.Format(L"%4.4f   ", channelR[index]);		f.WriteString(text);
			//text.Format(L"%4.4f   ", channelG[index]);		f.WriteString(text);
			//text.Format(L"%4.4f \n", channelB[index]);		f.WriteString(text);
			index++;
		}
	//f.Close();

	// ------ sau do dua 3 channels ve kieu matrix cua Matlab ------	
	mxArray *matlabChannelR, *matlabChannelG, *matlabChannelB;
	matlabChannelR = mxCreateDoubleMatrix(height, width, mxREAL);		memcpy((char *) mxGetPr(matlabChannelR), (char *) channelR, imgSize*sizeof(double));
	matlabChannelG = mxCreateDoubleMatrix(height, width, mxREAL);		memcpy((char *) mxGetPr(matlabChannelG), (char *) channelG, imgSize*sizeof(double));
	matlabChannelB = mxCreateDoubleMatrix(height, width, mxREAL);		memcpy((char *) mxGetPr(matlabChannelB), (char *) channelB, imgSize*sizeof(double));

	Engine *matlabEngine;
	matlabEngine = engOpen(NULL); // Start up MATLAB engine
	//engSetVisible(matlabEngine, false); // hide MATLAB engine session (neu ko co cai nay se bi hien Matlab command prompt, DUNG DE DEBUG)

	engPutVariable(matlabEngine, "matlabChannelR", matlabChannelR); // "matlabChannelR" la symbol (name) 
	engPutVariable(matlabEngine, "matlabChannelG", matlabChannelG);
	engPutVariable(matlabEngine, "matlabChannelB", matlabChannelB);

	// debug
	//double *a = mxGetPr(matlabChannelR);

	// ------ dua rX 1xN, rY 1xM ve kieu matrix cua Matlab ------

	// convert rX, rY tu kieu float* sang double* de co the dung voi kieu mxArray cua Matlab
	int rXWidth = rX->width;
	float *rXDataFloat = rX->data.fl;
	float *rYDataFloat = rY->data.fl;
	
	double *rXData = new double[rXWidth];
	double *rYData = new double[rXWidth];
	for (int i = 0; i < rX->width; i++) { 
		rXData[i] = rXDataFloat[i]; // double = float
		rYData[i] = rYDataFloat[i];
	}

	//
	mxArray *matlabRX, *matlabRY;
	matlabRX = mxCreateDoubleMatrix(1, rXWidth, mxREAL);		memcpy((char *) mxGetPr(matlabRX), (char *) rXData, rXWidth*sizeof(double));
	matlabRY = mxCreateDoubleMatrix(1, rXWidth, mxREAL);		memcpy((char *) mxGetPr(matlabRY), (char *) rYData, rXWidth*sizeof(double));

	engPutVariable(matlabEngine, "matlabRX", matlabRX);
	engPutVariable(matlabEngine, "matlabRY", matlabRY);

	// debug matlabRX, matlabRY
	//double *a = mxGetPr(matlabRX);
	//CStdioFile f;	f.Open(L"matlabR.txt", CFile::modeCreate | CFile::modeWrite);	CString text;
	//for (int i = 0; i < rXWidth; i++) { text.Format(L"%4.4f   ", a[i]);		f.WriteString(text); } f.Close();

	// ------ execute Matlab command ------	
	engEvalString(matlabEngine,"ansImgROIr = interp2(matlabChannelR, matlabRX, matlabRY, 'linear')");
	//engEvalString(matlabEngine, "err = lasterr");
	engEvalString(matlabEngine,"ansImgROIg = interp2(matlabChannelG, matlabRX, matlabRY, 'linear')");
	engEvalString(matlabEngine,"ansImgROIb = interp2(matlabChannelB, matlabRX, matlabRY, 'linear')");

	mxArray *ansImgROIr, *ansImgROIg, *ansImgROIb;
	ansImgROIr = engGetVariable(matlabEngine,"ansImgROIr");
	ansImgROIg = engGetVariable(matlabEngine,"ansImgROIg");
	ansImgROIb = engGetVariable(matlabEngine,"ansImgROIb");

	//mxArray *err = engGetVariable(matlabEngine,"err"); 
	//char *errChar = mxArrayToString(err); // convert mxArray to char *
	//CString errString(errChar); // convert char * to CString 

	// copy mxArray to CvMat (imgROIr, imgROIg, imgROIb)
	double *tmpR, *tmpG, *tmpB;
	tmpR = mxGetPr(ansImgROIr);
	tmpG = mxGetPr(ansImgROIg);
	tmpB = mxGetPr(ansImgROIb);
	for (int i = 0; i < rXWidth; i++) {
		CV_MAT_ELEM(*imgROIr, float, 0, i) = float(tmpR[i]);
		CV_MAT_ELEM(*imgROIg, float, 0, i) = float(tmpG[i]);
		CV_MAT_ELEM(*imgROIb, float, 0, i) = float(tmpB[i]);
	}

	// ------ release ------
	engClose(matlabEngine); // Shut down MATLAB engine

	delete channelR;
	delete channelG;
	delete channelB;

	mxDestroyArray(matlabChannelR);
	mxDestroyArray(matlabChannelG);
	mxDestroyArray(matlabChannelB);

	delete rXData;
	delete rYData;

	mxDestroyArray(matlabRX);
	mxDestroyArray(matlabRY);

	mxDestroyArray(ansImgROIr);
	mxDestroyArray(ansImgROIg);
	mxDestroyArray(ansImgROIb);*/
}
