
// CarScratchDetectorDlg.h : ��� ����
//

#pragma once
#include <opencv2\opencv.hpp>
#include "AlgorithmCode\AlgorithmCollection.h"
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

	cv::Mat m_currentMat;
	cv::Mat m_resultMat;


	// ������ �޽��� �� �Լ�
	virtual BOOL OnInitDialog();
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnBnClickedOpenimagefilebtn();
	CStatic m_imageWidthText;
	CStatic m_imageHeightText;
	CEdit m_spatialRadiusEditBox;
	CEdit m_colorRadiusEditBox;
	afx_msg void OnBnClickedClearbtn();

private:
	std::string m_currentFileName;
	ImageDescriptor m_currentImageDescriptor;
	std::map<std::string, ImageDescriptor> m_loadedImageDescriptorsMap;

public:
	afx_msg void OnCurrentImageTestButton();
	afx_msg void OnBnClickedLoaddbbutton();
	afx_msg void OnBnClickedTrainingsessionbutton();
};
