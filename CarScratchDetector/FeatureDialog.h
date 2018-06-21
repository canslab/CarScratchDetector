#pragma once
#include "afxwin.h"
#include "afxcmn.h"
#include "AlgorithmCode\AlgorithmCollection.h"

// CFeatureDialog dialog

class CFeatureDialog : public CDialogEx
{
	DECLARE_DYNAMIC(CFeatureDialog)

public:
	CFeatureDialog(CWnd* pParent = NULL);   // standard constructor
	virtual ~CFeatureDialog();

// Dialog Data
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_FEATUREDIALOG };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support
	void LoadFeatureData();

	DECLARE_MESSAGE_MAP()
public:
	std::vector<ImageDescriptor> m_featureData;

	CListCtrl m_featureListControl;
	virtual BOOL OnInitDialog();
};
