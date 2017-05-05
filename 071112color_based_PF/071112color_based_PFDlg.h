// 071112color_based_PFDlg.h : header file
//

#pragma once
#include "cv.h"
#include "highgui.h"
#include "GenericParticleFilter.h"
#include "ColorBasedPF.h"
#include "ColorBasedPF_xyVxVy.h"
#include "MeanShift.h"
#include "MeanShiftEMScale.h"
#include "MeanShiftEMScaleKalman.h"

// -----------------------------------------------

#define IMG_SEQ_TENNIS					"..\\testdata\\tennis\\"
#define IMG_SEQ_TENNIS_NUM_IMGS			(50)
#define IMG_SEQ_TENNIS_BALL_H			(6)
#define IMG_SEQ_TENNIS_BALL_POSITION	cvPoint(145, 63)

#define IMG_SEQ_BAD						"..\\testdata\\bad\\"
#define IMG_SEQ_BAD_NUM_IMGS			(50)
#define IMG_SEQ_BAD_WHITE_CAR_H			(8)
#define IMG_SEQ_BAD_WHITE_CAR_POSITION	cvPoint(113, 84)

#define IMG_SEQ_DTNEUSCHNEE				"..\\testdata\\dtneu_schnee\\"
#define IMG_SEQ_DTNEUSCHNEE_NUM_IMGS			(100)
#define IMG_SEQ_DTNEUSCHNEE_RED_CAR_H			(8)
#define IMG_SEQ_DTNEUSCHNEE_RED_CAR_POSITION	cvPoint(97, 47)


#define IMG_TYPE						".BMP"

// -----------------------------------------------

#define WIN_NAME_WEBCAM			"Webcam"
#define WIN_NAME_IMAGE_SEQUENCE "Image Sequence"

#define DELAY_TIME				(100)

// particle filter params
#define NUM_OF_PARTICLES	(2000)
#define POSITION_SIGMA		(32.0)	// for random number generator
#define VELOCITY_SIGMA		(8.0)	// for random number generator
#define SIZE_SIGMA			(1.0)	// for random number generator

// mean shift params
#define MS_WIN_WIDTH		(80)
#define MS_WIN_HEIGHT		(80)
#define MS_EPSILON			(1.0)

// -----------------------------------------------

// CMy071112color_based_PFDlg dialog
class CMy071112color_based_PFDlg : public CDialog
{
// Construction
public:
	CMy071112color_based_PFDlg(CWnd* pParent = NULL);	// standard constructor

// Dialog Data
	enum { IDD = IDD_MY071112COLOR_BASED_PF_DIALOG };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support


// Implementation
protected:
	HICON m_hIcon;

	// Generated message map functions
	virtual BOOL OnInitDialog();
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()


private:
	void getUserInput();
	void displayResult();

public:
	afx_msg void OnBnClickedButtonImageSequence();
public:
	afx_msg void OnBnClickedButtonVideo();
public:
	afx_msg void OnBnClickedButtonCameraMatrox();
public:
	afx_msg void OnBnClickedButtonWebcam();
public:
	afx_msg void OnBnClickedButtonQuit();
public:
	afx_msg void OnBnClickedButtonStop();
public:
	afx_msg void OnBnClickedButtonNextImg();
public:
	afx_msg void OnBnClickedButtonLoadImg();
public:
	afx_msg void OnBnClickedButtonDefineObj();
};
