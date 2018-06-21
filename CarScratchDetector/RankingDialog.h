#pragma once


// RankingDialog dialog

#include <vector>
#include <string>
#include "afxcmn.h"
#include <map>

class RankingDialog : public CDialogEx
{
	DECLARE_DYNAMIC(RankingDialog)

public:
	RankingDialog(CWnd* pParent = NULL);   // standard constructor
	virtual ~RankingDialog();

// Dialog Data
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_RANKDIALOG };
#endif

public:
	std::vector<std::string> m_fileNames;
	std::vector<double> m_distanceRecord;

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support

	DECLARE_MESSAGE_MAP()
public:
	virtual BOOL OnInitDialog();
	CListCtrl m_list;
};
