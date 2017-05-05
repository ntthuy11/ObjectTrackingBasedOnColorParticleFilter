// 071112color_based_PF.h : main header file for the PROJECT_NAME application
//

#pragma once

#ifndef __AFXWIN_H__
	#error "include 'stdafx.h' before including this file for PCH"
#endif

#include "resource.h"		// main symbols


// CMy071112color_based_PFApp:
// See 071112color_based_PF.cpp for the implementation of this class
//

class CMy071112color_based_PFApp : public CWinApp
{
public:
	CMy071112color_based_PFApp();

// Overrides
	public:
	virtual BOOL InitInstance();

// Implementation

	DECLARE_MESSAGE_MAP()
};

extern CMy071112color_based_PFApp theApp;