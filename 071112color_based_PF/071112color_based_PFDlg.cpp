// 071112color_based_PFDlg.cpp : implementation file
#include "stdafx.h"
#include "071112color_based_PF.h"
#include "071112color_based_PFDlg.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

// ----------------------------
ColorBasedPF_xyVxVy colorPF_xyVxVy;
ColorBasedPF colorPF;
MeanShift meanShift;
MeanShiftEMScale meanShiftEMScale;
MeanShiftEMScaleKalman meanShiftEMScaleKalman;

CStringA currImgSeq;
int currImgSeq_numImgs;

bool running = true, autoRun = false;
unsigned long TimeDiff;

int numParticles, hx, hy, imgIndex = 0, msWinWidth, msWinHeight;
double positionSigma, velocitySigma, sizeSigma, msEpsilon;

IplImage *currFrame = 0, *tmpFrame = 0;

CvPoint previousPoint = cvPoint(-1, -1);
CvPoint currentPoint = cvPoint(-1, -1);
CvPoint centralPoint = cvPoint(-1, -1);

// ----------------------------

void onMouseDefineObjPosition(int event, int x, int y, int flags, void* notUsed) {
	currentPoint = cvPoint(x,y);

	switch (event) {

		case CV_EVENT_LBUTTONDOWN: 
			previousPoint = cvPoint(x,y);
			break;

		case CV_EVENT_MOUSEMOVE: 
			if (previousPoint.x != -1) {
				cvCopyImage(currFrame, tmpFrame);
				cvRectangle(tmpFrame, previousPoint, currentPoint, CV_RGB(0,0,255));
			}
			cvShowImage(WIN_NAME_IMAGE_SEQUENCE, tmpFrame);
			break;

		case CV_EVENT_RBUTTONDOWN: 
			if (previousPoint.x != -1) cvRectangle(tmpFrame, previousPoint, currentPoint, CV_RGB(0,0,255));			
			cvShowImage(WIN_NAME_IMAGE_SEQUENCE, tmpFrame);	

			//
			centralPoint = cvPoint((previousPoint.x + currentPoint.x) / 2, (previousPoint.y + currentPoint.y) / 2);
			hx = abs(centralPoint.x - previousPoint.x);
			hy = abs(centralPoint.y - previousPoint.y);

			previousPoint = cvPoint(-1, -1);			
			break;
	}
}

// ----------------------------

CMy071112color_based_PFDlg::CMy071112color_based_PFDlg(CWnd* pParent /*=NULL*/)
	: CDialog(CMy071112color_based_PFDlg::IDD, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CMy071112color_based_PFDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CMy071112color_based_PFDlg, CDialog)
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	//}}AFX_MSG_MAP
	ON_BN_CLICKED(IDC_BUTTON_IMAGE_SEQUENCE, &CMy071112color_based_PFDlg::OnBnClickedButtonImageSequence)
	ON_BN_CLICKED(IDC_BUTTON_VIDEO, &CMy071112color_based_PFDlg::OnBnClickedButtonVideo)
	ON_BN_CLICKED(IDC_BUTTON_CAMERA_MATROX, &CMy071112color_based_PFDlg::OnBnClickedButtonCameraMatrox)
	ON_BN_CLICKED(IDC_BUTTON_WEBCAM, &CMy071112color_based_PFDlg::OnBnClickedButtonWebcam)
	ON_BN_CLICKED(IDC_BUTTON_QUIT, &CMy071112color_based_PFDlg::OnBnClickedButtonQuit)
	ON_BN_CLICKED(IDC_BUTTON_STOP, &CMy071112color_based_PFDlg::OnBnClickedButtonStop)
	ON_BN_CLICKED(IDC_BUTTON_NEXT_IMG, &CMy071112color_based_PFDlg::OnBnClickedButtonNextImg)
	ON_BN_CLICKED(IDC_BUTTON_LOAD_IMG, &CMy071112color_based_PFDlg::OnBnClickedButtonLoadImg)
	ON_BN_CLICKED(IDC_BUTTON_DEFINE_OBJ, &CMy071112color_based_PFDlg::OnBnClickedButtonDefineObj)
END_MESSAGE_MAP()

// CMy071112color_based_PFDlg message handlers

BOOL CMy071112color_based_PFDlg::OnInitDialog()
{
	CDialog::OnInitDialog();

	// Set the icon for this dialog.  The framework does this automatically
	//  when the application's main window is not a dialog
	SetIcon(m_hIcon, TRUE);			// Set big icon
	SetIcon(m_hIcon, FALSE);		// Set small icon

	// TODO: Add extra initialization here

	//--------------------- INIT --------------------------
	CString s;
	s.Format(L"%4d", NUM_OF_PARTICLES);		this->GetDlgItem(IDC_EDIT_NUM_OF_PARTICLES)->SetWindowTextW((LPCTSTR)s); 
	s.Format(L"%3.2f", POSITION_SIGMA);		this->GetDlgItem(IDC_EDIT_POSITION_SIGMA)->SetWindowTextW((LPCTSTR)s); 
	s.Format(L"%3.2f", VELOCITY_SIGMA);		this->GetDlgItem(IDC_EDIT_VELOCITY_SIGMA)->SetWindowTextW((LPCTSTR)s); 
	s.Format(L"%3.2f", SIZE_SIGMA);			this->GetDlgItem(IDC_EDIT_SIZE_SIGMA)->SetWindowTextW((LPCTSTR)s); 

	s.Format(L"%3d", MS_WIN_WIDTH);			this->GetDlgItem(IDC_EDIT_MS_WIN_WIDTH)->SetWindowTextW((LPCTSTR)s); 
	s.Format(L"%3d", MS_WIN_HEIGHT);		this->GetDlgItem(IDC_EDIT_MS_WIN_HEIGHT)->SetWindowTextW((LPCTSTR)s); 
	s.Format(L"%3.2f", MS_EPSILON);			this->GetDlgItem(IDC_EDIT_MS_EPSILON)->SetWindowTextW((LPCTSTR)s); 

	CComboBox *combo;
	combo = (CComboBox *) this->GetDlgItem(IDC_COMBO_ALGORITHM);
	combo->AddString(L"Color-based PF (x, y, vx, vy)");			// 0	=> ColorBasedPF_xyVxVy.cpp
	combo->AddString(L"Color-based PF (x, y, vx, vy, hx, hy) [NOT USED]");	// 1	=> ColorBasedPF.cpp
	//combo->AddString(L"Generic PF"); // 
	combo->AddString(L"Mean Shift");							// 2	=> MeanShift.cpp
	combo->AddString(L"Mean Shift (EM Scale) [NOT USED]");					// 3	=> MeanShiftEMScale.cpp
	combo->AddString(L"Mean Shift (EM Scale, Kalman)");			// 4	=> MeanShiftEMScaleKalman.cpp

	combo->SetCurSel(0);

	combo = (CComboBox *) this->GetDlgItem(IDC_COMBO_IMG_SEQUENCE);
	combo->AddString(L"bad"); // 0
	combo->AddString(L"dtneu_schnee"); // 1
	combo->AddString(L"tennis"); // 2 
	combo->SetCurSel(2);

	// ----------------------------------------------------

	return TRUE;  // return TRUE  unless you set the focus to a control
}

// If you add a minimize button to your dialog, you will need the code below
//  to draw the icon.  For MFC applications using the document/view model,
//  this is automatically done for you by the framework.

void CMy071112color_based_PFDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // device context for painting

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// Center icon in client rectangle
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// Draw the icon
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialog::OnPaint();
	}
}

// The system calls this function to obtain the cursor to display while the user drags
//  the minimized window.
HCURSOR CMy071112color_based_PFDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}

void CMy071112color_based_PFDlg::OnBnClickedButtonImageSequence() {	
	autoRun = true;

	getUserInput();
	
	// load the first image (0.bmp)
	CStringA firstImgFilename;
	firstImgFilename.Append(currImgSeq);	firstImgFilename.Append("0");	firstImgFilename.Append(IMG_TYPE); 
	currFrame = cvvLoadImage(firstImgFilename);
	cvNamedWindow(WIN_NAME_IMAGE_SEQUENCE, CV_WINDOW_AUTOSIZE);	
	cvShowImage(WIN_NAME_IMAGE_SEQUENCE, currFrame);

	// define object
	OnBnClickedButtonDefineObj();

	// process other images
	while (imgIndex < currImgSeq_numImgs && running) {
		OnBnClickedButtonNextImg();
		cvWaitKey(DELAY_TIME);
	}

	//
	autoRun = false;
	imgIndex = 0;
}

void CMy071112color_based_PFDlg::OnBnClickedButtonVideo() {
	
}

void CMy071112color_based_PFDlg::OnBnClickedButtonCameraMatrox() {
	
}

void CMy071112color_based_PFDlg::OnBnClickedButtonWebcam() {
/*	CvCapture* capture = cvCaptureFromCAM(0); // capture from video device #0
		
	if(!capture) {
		MessageBox(L"Could not initialize capturing.");
	} else { 	
		getUserInput(); 
		cvNamedWindow(WIN_NAME_WEBCAM, CV_WINDOW_AUTOSIZE);

		currFrame = cvQueryFrame(capture); // frame dau tien

		//
		switch (((CComboBox *) GetDlgItem(IDC_COMBO_ALGORITHM))->GetCurSel()) {

			// ---------- Color-based PF ----------
			case 0: { 
				
				
				break;
					}

			// ---------- point cloud algo ----------
			case 1:	{ 
				GenericParticleFilter genericPF(currFrame, numParticles, SIGMA);
			
				while(running) {
					currFrame = cvQueryFrame(capture);
					genericPF.process(currFrame);	
					cvShowImage(WIN_NAME_WEBCAM, currFrame);
					cvWaitKey(5); // delay 1 khoang thoi gian moi co the show image len duoc
				}

				break;
					}
		}

		//cvReleaseImage(&currFrame); // ko biet vi sao release lai bi loi			
	}
	cvReleaseCapture(&capture);
	*/
}

void CMy071112color_based_PFDlg::OnBnClickedButtonLoadImg() {
	getUserInput();

	// load the first image (0.bmp)
	CStringA firstImgFilename;
	firstImgFilename.Append(currImgSeq);	firstImgFilename.Append("0");	firstImgFilename.Append(IMG_TYPE); 

	currFrame = cvvLoadImage(firstImgFilename);
	tmpFrame = cvCreateImage(cvSize(currFrame->width, currFrame->height), currFrame->depth, currFrame->nChannels);
	cvCopyImage(currFrame, tmpFrame); // dung de ve hinh chu nhat nham xac dinh obj

	// show image
	cvNamedWindow(WIN_NAME_IMAGE_SEQUENCE, CV_WINDOW_AUTOSIZE);	
	cvShowImage(WIN_NAME_IMAGE_SEQUENCE, currFrame);

	// set mouse event
	cvSetMouseCallback(WIN_NAME_IMAGE_SEQUENCE, onMouseDefineObjPosition, &currFrame);
}

void CMy071112color_based_PFDlg::OnBnClickedButtonDefineObj() {
	switch (((CComboBox *) GetDlgItem(IDC_COMBO_ALGORITHM))->GetCurSel()) {
		case 0: //Color-based PF (x, y, vx, vy)
			colorPF_xyVxVy.setParamsAndInit(numParticles, hx, hy, centralPoint, positionSigma, velocitySigma, currFrame);
			break;

		case 1: //Color-based PF (x, y, vx, vy, hx, hy)
			colorPF.setParamsAndInit(numParticles, hx, hy, centralPoint, positionSigma, velocitySigma, sizeSigma, currFrame);
			break;

		case 2:
			meanShift.setParamsAndInit(hx, hy, msWinWidth, msWinHeight, msEpsilon, centralPoint, currFrame);
			break;

		case 3:
			meanShiftEMScale.setParamsAndInit(hx, hy, msWinWidth, msWinHeight, msEpsilon, centralPoint, currFrame);
			break;

		case 4:
			meanShiftEMScaleKalman.setParamsAndInit(hx, hy, msWinWidth, msWinHeight, msEpsilon, centralPoint, currFrame);
			break;
	}	
}

void CMy071112color_based_PFDlg::OnBnClickedButtonNextImg() {
	
	// ----------------- tick count START -----------------
	__int64 freq, tStart, tStop;	
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq); // Get the frequency of the hi-res timer
	QueryPerformanceCounter((LARGE_INTEGER*)&tStart); // Assuming that has worked you can then use hi-res timer 
	// ----------------------------------------------------

	if (imgIndex < currImgSeq_numImgs) {
		// load image
		CStringA fn;		fn.Format("%1d", imgIndex);
		CStringA filename;	filename.Append(currImgSeq);	filename.Append(fn);	filename.Append(IMG_TYPE);
		currFrame = cvvLoadImage(filename);

		//IplImage* graySrcImg = cvCreateImage(cvSize(currFrame->width, currFrame->height), IPL_DEPTH_8U, 1);
		//cvConvertImage(currFrame, graySrcImg);
		//cvCanny(graySrcImg, graySrcImg, 50, 200);
		//cvConvertImage(graySrcImg, currFrame);

		// show the result
		switch (((CComboBox *) GetDlgItem(IDC_COMBO_ALGORITHM))->GetCurSel()) {
			case 0: //Color-based PF (x, y, vx, vy)
				colorPF_xyVxVy.process(currFrame);
				break;

			case 1: //Color-based PF (x, y, vx, vy, hx, hy)
				colorPF.process(currFrame);
				break;

			case 2:
				meanShift.process(currFrame);
				break;

			case 3:
				meanShiftEMScale.process(currFrame);
				break;

			case 4:
				meanShiftEMScaleKalman.process(currFrame);
				break;
		}
		
		cvShowImage(WIN_NAME_IMAGE_SEQUENCE, currFrame);

		// save result
		/*CString saveFn;		saveFn.Format(L"..\\testdata\\bad_result_meanshift\\%2d.bmp", imgIndex);
		CStringA saveFilename (saveFn);
		cvSaveImage(saveFilename, currFrame);*/

		//
		imgIndex++;
		cvReleaseImage(&currFrame);
	}

	// ----------------- tick count STOP -----------------	
	QueryPerformanceCounter((LARGE_INTEGER*)&tStop); // Perform operations that require timing
	TimeDiff = (unsigned long)(((tStop - tStart) * 1000) / freq); // Calculate time difference
	// ----------------------------------------------------
	displayResult();
}

void CMy071112color_based_PFDlg::OnBnClickedButtonQuit() {
	running = false;

	cvDestroyAllWindows();
	cvReleaseImage(&currFrame);	
	cvReleaseImage(&tmpFrame);	

	OnOK();
}

void CMy071112color_based_PFDlg::OnBnClickedButtonStop() {	
	running = false;

	cvDestroyAllWindows();
	cvReleaseImage(&currFrame);
	cvReleaseImage(&tmpFrame);

	imgIndex = 0;
}

// ----------------------------------------------------------
void CMy071112color_based_PFDlg::getUserInput() {
	running = true; // <----

	// ------------------- edit box -------------------

	CString numOfParticlesString;	this->GetDlgItem(IDC_EDIT_NUM_OF_PARTICLES)->GetWindowTextW(numOfParticlesString);
	CString positionSigmaString;	this->GetDlgItem(IDC_EDIT_POSITION_SIGMA)->GetWindowTextW(positionSigmaString);
	CString velocitySigmaString;	this->GetDlgItem(IDC_EDIT_VELOCITY_SIGMA)->GetWindowTextW(velocitySigmaString);
	CString sizeSigmaString;		this->GetDlgItem(IDC_EDIT_VELOCITY_SIGMA)->GetWindowTextW(sizeSigmaString);
	
	CString msWinWidthString;		this->GetDlgItem(IDC_EDIT_MS_WIN_WIDTH)->GetWindowTextW(msWinWidthString);
	CString msWinHeightString;		this->GetDlgItem(IDC_EDIT_MS_WIN_HEIGHT)->GetWindowTextW(msWinHeightString);
	CString msEpsilonString;		this->GetDlgItem(IDC_EDIT_MS_EPSILON)->GetWindowTextW(msEpsilonString);

	numParticles = NUM_OF_PARTICLES;
	positionSigma = POSITION_SIGMA;
	velocitySigma = VELOCITY_SIGMA;
	sizeSigma = SIZE_SIGMA;
	
	msWinWidth = MS_WIN_WIDTH;
	msWinHeight = MS_WIN_HEIGHT;
	msEpsilon = MS_EPSILON;

	if (!numOfParticlesString.IsEmpty())	numParticles = (int)wcstod(numOfParticlesString, NULL);
	if (!positionSigmaString.IsEmpty())		positionSigma = wcstod(positionSigmaString, NULL);
	if (!velocitySigmaString.IsEmpty())		velocitySigma = wcstod(velocitySigmaString, NULL);
	if (!sizeSigmaString.IsEmpty())			sizeSigma = wcstod(sizeSigmaString, NULL);

	if (!msWinWidthString.IsEmpty())		msWinWidth = (int)wcstod(msWinWidthString, NULL);
	if (!msWinHeightString.IsEmpty())		msWinHeight = (int)wcstod(msWinHeightString, NULL);
	if (!msEpsilonString.IsEmpty())			msEpsilon = wcstod(msEpsilonString, NULL);

	// ------------- combo box --------------------

	switch (((CComboBox *) GetDlgItem(IDC_COMBO_IMG_SEQUENCE))->GetCurSel()) {

		case 0: // bad
			currImgSeq = IMG_SEQ_BAD;
			currImgSeq_numImgs = IMG_SEQ_BAD_NUM_IMGS;

			if (autoRun) {
				hx = hy = IMG_SEQ_BAD_WHITE_CAR_H;
				centralPoint = IMG_SEQ_BAD_WHITE_CAR_POSITION;
			}
			break;

		case 1: // dtneu_schnee
			currImgSeq = IMG_SEQ_DTNEUSCHNEE;
			currImgSeq_numImgs = IMG_SEQ_DTNEUSCHNEE_NUM_IMGS;

			if (autoRun) {
				hx = hy = IMG_SEQ_DTNEUSCHNEE_RED_CAR_H;
				centralPoint = IMG_SEQ_DTNEUSCHNEE_RED_CAR_POSITION;
			}
			break;

		case 2: // tennis
			currImgSeq = IMG_SEQ_TENNIS;
			currImgSeq_numImgs = IMG_SEQ_TENNIS_NUM_IMGS;

			if (autoRun) {
				hx = hy = IMG_SEQ_TENNIS_BALL_H;
				centralPoint = IMG_SEQ_TENNIS_BALL_POSITION;
			}
			break;
	}
	
}

void CMy071112color_based_PFDlg::displayResult() {
	CString outTextString;

	outTextString.Format(L"%5d", TimeDiff);
	GetDlgItem(IDC_STATIC_EXE_TIME)->SetWindowTextW((LPCTSTR)outTextString); 
}