
// CarScratchDetectorDlg.h : 헤더 파일
//

#pragma once
#include <opencv2\opencv.hpp>
#include "afxwin.h"

// CCarScratchDetectorDlg 대화 상자
class CCarScratchDetectorDlg : public CDialogEx
{
// 생성입니다.
public:
	CCarScratchDetectorDlg(CWnd* pParent = NULL);	// 표준 생성자입니다.

// 대화 상자 데이터입니다.
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_CARSCRATCHDETECTOR_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV 지원입니다.


// 구현입니다.
protected:
	HICON m_hIcon;

	cv::Mat m_srcMat;
	cv::Mat m_resultMat;
	

	// 생성된 메시지 맵 함수
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
