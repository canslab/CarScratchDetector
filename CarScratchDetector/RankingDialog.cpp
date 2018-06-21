// RankingDialog.cpp : implementation file
//

#include "stdafx.h"
#include "CarScratchDetector.h"
#include "RankingDialog.h"
#include "afxdialogex.h"


// RankingDialog dialog

IMPLEMENT_DYNAMIC(RankingDialog, CDialogEx)

RankingDialog::RankingDialog(CWnd* pParent /*=NULL*/)
	: CDialogEx(IDD_RANKDIALOG, pParent)
{

}

RankingDialog::~RankingDialog()
{
}

void RankingDialog::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_RANKLIST, m_list);
}


BEGIN_MESSAGE_MAP(RankingDialog, CDialogEx)
END_MESSAGE_MAP()


// RankingDialog message handlers


BOOL RankingDialog::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// TODO:  Add extra initialization here
	m_list.DeleteAllItems();
	m_list.SetExtendedStyle(LVS_EX_GRIDLINES);

	// TODO:  Add extra initialization here
	m_list.InsertColumn(0, _T("File Name"), LVCFMT_LEFT, 850, -1);
	m_list.InsertColumn(1, _T("Distance"), LVCFMT_LEFT, 120, -1);

	int kTotalCount = (m_fileNames.size() <= 5) ? m_fileNames.size() : 5;

	for(int i = 0; i < kTotalCount; ++i)
	{
		m_list.InsertItem(i, m_fileNames[i].c_str());
		m_list.SetItem(i, 1, LVIF_TEXT, _T(std::to_string(m_distanceRecord[i]).c_str()), 0, 0, 0, NULL);
	}

	return TRUE;  // return TRUE unless you set the focus to a control
				  // EXCEPTION: OCX Property Pages should return FALSE
}
