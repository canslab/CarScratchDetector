#pragma once


// TrainingDialog dialog
#include <opencv2\opencv.hpp>
#include "AlgorithmCode\AlgorithmCollection.h"
#include "afxcmn.h"

class TrainingDialog : public CDialogEx
{
	DECLARE_DYNAMIC(TrainingDialog)

public:
	TrainingDialog(CWnd* pParent = NULL);   // standard constructor
	virtual ~TrainingDialog();

	// Dialog Data
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_TRAININGDIALOG };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support
	std::string GetImageFileNameFromPath(CString in_path); // Utility Function

	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnBnClickedOpenimagebutton();
	afx_msg void OnBnClickedDiscardbutton();
	afx_msg void OnBnClickedExportbutton();
	virtual BOOL OnInitDialog();


	CListCtrl m_fileNameList;
	std::map<std::string, ImageDescriptor> m_imageDescriptorsMap;
};
