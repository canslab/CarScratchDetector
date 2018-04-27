
// CarScratchDetectorDlg.h : ��� ����
//

#pragma once
#include <opencv2\opencv.hpp>
#include "afxwin.h"

// CCarScratchDetectorDlg ��ȭ ����
class CCarScratchDetectorDlg : public CDialogEx
{
// �����Դϴ�.
public:
	CCarScratchDetectorDlg(CWnd* pParent = NULL);	// ǥ�� �������Դϴ�.

// ��ȭ ���� �������Դϴ�.
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_CARSCRATCHDETECTOR_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV �����Դϴ�.


// �����Դϴ�.
protected:
	HICON m_hIcon;

	cv::Mat m_srcMat;
	cv::Mat m_resultMat;
	

	// ������ �޽��� �� �Լ�
	virtual BOOL OnInitDialog();
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnBnClickedOpenimagefilebtn();
	afx_msg void OnBnClickedShowbodypart();
	afx_msg void OnBnClickedRunbutton();
	CStatic m_imageWidthText;
	CStatic m_imageHeightText;
	afx_msg void OnBnClickedSaveresultbtn();
	CEdit m_spatialRadiusEditBox;
	CEdit m_colorRadiusEditBox;
	CButton m_bCheckEdgeMap;
	CButton m_bCheckLabelMap;
	CButton m_bCheckCornerMap;
	afx_msg void OnBnClickedAnalyze();
};
