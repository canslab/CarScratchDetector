
// CarScratchDetector.h : PROJECT_NAME ���� ���α׷��� ���� �� ��� �����Դϴ�.
//

#pragma once

#ifndef __AFXWIN_H__
	#error "PCH�� ���� �� ������ �����ϱ� ���� 'stdafx.h'�� �����մϴ�."
#endif

#include "resource.h"		// �� ��ȣ�Դϴ�.


// CCarScratchDetectorApp:
// �� Ŭ������ ������ ���ؼ��� CarScratchDetector.cpp�� �����Ͻʽÿ�.
//

class CCarScratchDetectorApp : public CWinApp
{
public:
	CCarScratchDetectorApp();

// �������Դϴ�.
public:
	virtual BOOL InitInstance();

// �����Դϴ�.

	DECLARE_MESSAGE_MAP()
};

extern CCarScratchDetectorApp theApp;