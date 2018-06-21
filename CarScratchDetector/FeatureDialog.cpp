// FeatureDialog.cpp : implementation file
//

#include "stdafx.h"
#include "CarScratchDetector.h"
#include "FeatureDialog.h"
#include "afxdialogex.h"


// CFeatureDialog dialog

IMPLEMENT_DYNAMIC(CFeatureDialog, CDialogEx)

CFeatureDialog::CFeatureDialog(CWnd* pParent /*=NULL*/)
	: CDialogEx(IDD_FEATUREDIALOG, pParent)
{


}

CFeatureDialog::~CFeatureDialog()
{

}

void CFeatureDialog::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	//  DDX_Control(pDX, IDC_LIST2, m_feaureList);
	DDX_Control(pDX, IDC_LIST1, m_featureListControl);
}


BEGIN_MESSAGE_MAP(CFeatureDialog, CDialogEx)
END_MESSAGE_MAP()


// CFeatureDialog message handlers

void CFeatureDialog::LoadFeatureData()
{
}

BOOL CFeatureDialog::OnInitDialog()
{
	CDialogEx::OnInitDialog();
	m_featureListControl.DeleteAllItems();
	m_featureListControl.SetExtendedStyle(LVS_EX_GRIDLINES);
	
	// TODO:  Add extra initialization here
	m_featureListControl.InsertColumn(0, _T("File Name"), LVCFMT_LEFT, 140, -1);
	m_featureListControl.InsertColumn(1, _T("# of Points"), LVCFMT_LEFT, 140, -1);
	m_featureListControl.InsertColumn(2, _T("Mean Position"), LVCFMT_CENTER, 109, -1);
	m_featureListControl.InsertColumn(3, _T("Stddev"), LVCFMT_CENTER, 109, -1);
	m_featureListControl.InsertColumn(4, _T("Skewness"), LVCFMT_CENTER, 109, -1);
	m_featureListControl.InsertColumn(5, _T("Density(ROI)"), LVCFMT_CENTER, 109, -1);
	m_featureListControl.InsertColumn(6, _T("Density(Effective)"), LVCFMT_CENTER, 109, -1);
	m_featureListControl.InsertColumn(7, _T("# of Points (LCluster)"), LVCFMT_CENTER, 130, -1);
	m_featureListControl.InsertColumn(8, _T("Mean Position (LCluster)"), LVCFMT_CENTER, 130, -1);
	m_featureListControl.InsertColumn(9, _T("Stddev (LCluster)"), LVCFMT_CENTER, 109, -1);
	m_featureListControl.InsertColumn(10, _T("Skewness (LCluster)"), LVCFMT_CENTER, 109, -1);
	m_featureListControl.InsertColumn(11, _T("# of Dense Clusters"), LVCFMT_CENTER, 109, -1);
	m_featureListControl.InsertColumn(12, _T("Eigvalue Ratio (LCluster)"), LVCFMT_CENTER, 109, -1);
	m_featureListControl.InsertColumn(13, _T("Largest Eig value (LCluster)"), LVCFMT_CENTER, 109, -1);
	m_featureListControl.InsertColumn(14, _T("Small Eig value (LCluster)"), LVCFMT_CENTER, 109, -1);
	m_featureListControl.InsertColumn(15, _T("Orientation (LCluster)"), LVCFMT_CENTER, 109, -1);

	LoadFeatureData();
	return TRUE;  // return TRUE unless you set the focus to a control
				  // EXCEPTION: OCX Property Pages should return FALSE
}
