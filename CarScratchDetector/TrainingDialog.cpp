// TrainingDialog.cpp : implementation file
//

#include "stdafx.h"
#include "CarScratchDetector.h"
#include "TrainingDialog.h"
#include "afxdialogex.h"


// TrainingDialog dialog

IMPLEMENT_DYNAMIC(TrainingDialog, CDialogEx)

TrainingDialog::TrainingDialog(CWnd* pParent /*=NULL*/)
	: CDialogEx(IDD_TRAININGDIALOG, pParent)
{

}

TrainingDialog::~TrainingDialog()
{
}

void TrainingDialog::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_LIST1, m_fileNameList);
}

std::string TrainingDialog::GetImageFileNameFromPath(CString in_path)
{
	std::string temp = in_path.GetString();

	auto tt = temp.find(".jpg");
	for (int i = tt; i >= 0; --i)
	{
		char k = temp.at(i);
		if (k == '\\')
		{
			tt = i + 1;
			break;
		}
	}
	auto leee = temp.length() - tt;
	auto currentFileTitle = temp.substr(tt, leee);

	return currentFileTitle;
}


BEGIN_MESSAGE_MAP(TrainingDialog, CDialogEx)
	ON_BN_CLICKED(IDC_OPENIMAGEBUTTON, &TrainingDialog::OnBnClickedOpenimagebutton)
	ON_BN_CLICKED(IDC_DISCARDBUTTON, &TrainingDialog::OnBnClickedDiscardbutton)
	ON_BN_CLICKED(IDC_EXPORTBUTTON, &TrainingDialog::OnBnClickedExportbutton)
END_MESSAGE_MAP()


// TrainingDialog message handlers

void TrainingDialog::OnBnClickedOpenimagebutton()
{
	char szFilter[] = "Image|*.BMP;*.PNG;*.JPG;*.JPEG";
	CFileDialog dlg(TRUE, NULL, NULL, OFN_HIDEREADONLY | OFN_OVERWRITEPROMPT | OFN_FILEMUSTEXIST | OFN_ALLOWMULTISELECT, szFilter, AfxGetMainWnd());
	//dlg.GetStartPosition();

	POSITION pos(dlg.GetStartPosition());

	if (dlg.DoModal() == IDOK)
	{
		while (pos)
		{

			CString currentFilePath = dlg.GetNextPathName(pos);
			std::string currentFileName = GetImageFileNameFromPath(currentFilePath);
			cv::Mat currentMat;

			std::string statusString = currentFileName + " 처리중";
			GetDlgItem(IDC_STATUSTEXT)->SetWindowTextA(statusString.c_str());

			currentMat = cv::imread(cv::String(currentFilePath).c_str());
			ResizeImageUsingThreshold(currentMat, 100000);

			if (currentMat.data != nullptr)
			{
				cv::Mat carBodyBinaryImage;
				cv::Rect boundingBox;
				std::vector<cv::Point> carBodyContourPoints;
				std::vector<cv::Point2f> scratchPoints;
				std::vector<Cluster_DBSCAN> scratchClusters;
				m_imageDescriptorsMap[currentFileName] = ImageDescriptor();
				ExtractImageDescriptorFromMat(currentMat, m_imageDescriptorsMap[currentFileName]);
				m_fileNameList.InsertItem(m_imageDescriptorsMap.size() - 1, _T(currentFileName.c_str()));
			}
		}
	}

	// TODO: Add your control notification handler code here
	GetDlgItem(IDC_STATUSTEXT)->SetWindowTextA("Image 처리 완료");
	GetDlgItem(IDC_OPENIMAGEBUTTON)->EnableWindow(FALSE);
	GetDlgItem(IDC_DISCARDBUTTON)->EnableWindow(TRUE);
}

void TrainingDialog::OnBnClickedDiscardbutton()
{
	GetDlgItem(IDC_STATUSTEXT)->SetWindowTextA("기록 삭제완료");
	m_fileNameList.DeleteAllItems();
	m_imageDescriptorsMap.clear();
}


void TrainingDialog::OnBnClickedExportbutton()
{
	FILE* fp = fopen("features.txt", "w");

	fprintf(fp, "Total#\tMean Position(x,y)\tStd dev(x, y)\tSkewness\tDensity(ROI)\tDensity(Effective)\tTotal #(Largest)\tMean Position(Largest)\tStd dev(Largest)\tSkewness(Largest)\t# of Dense Clusters\t");
	fprintf(fp, "LargestCluster Eigvalue ratio\tLargestCluster large Eigenvalue\tLargestCluster small Eigenvalue\tLargestCluster Orientation\n");

	for (auto& eachDescriptor : m_imageDescriptorsMap)
	{
		fprintf(fp, "%6s\t%4d\t%4.2f\t%4.2f\t%4.2f\t%4.2f\t%4.2f\t%4.2f\t%4.2f\t%4d\t%4.2f\t%4.2f\t%4.2f\t%4.2f\t%4.2f\t%4d\t%4.2f\t%4.2f\t%4.2f\t%4.2f\n",
			eachDescriptor.first.c_str(),

			eachDescriptor.second.m_totalNumberOfPointsInROI,
			eachDescriptor.second.m_globalMeanPosition.x,
			eachDescriptor.second.m_globalMeanPosition.y,
			eachDescriptor.second.m_globalStdDev[0],
			eachDescriptor.second.m_globalStdDev[1],
			eachDescriptor.second.m_globalSkewness,
			eachDescriptor.second.m_globalDensityInROI,
			eachDescriptor.second.m_globalDensityInEffectiveROI,
			eachDescriptor.second.m_totalNumberOfPointsOfLargestCluster,
			eachDescriptor.second.m_largestClusterMeanPosition.x,
			eachDescriptor.second.m_largestClusterMeanPosition.y,
			eachDescriptor.second.m_largestClusterStdDev[0],
			eachDescriptor.second.m_largestClusterStdDev[1],
			eachDescriptor.second.m_largestClusterSkewness,
			eachDescriptor.second.m_numberOfDenseClusters,

			eachDescriptor.second.m_largestClusterEigenvalueRatio,
			eachDescriptor.second.m_largestClusterLargeEigenValue,
			eachDescriptor.second.m_largestClusterSmallEigenValue,
			eachDescriptor.second.m_largestClusterOrientation
		);
	}

	fclose(fp);
	MessageBox("features.txt로 프로젝트 디렉토리에 저장하였습니다");
	GetDlgItem(IDC_STATUSTEXT)->SetWindowTextA("DB Export 완료");
}


BOOL TrainingDialog::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// TODO:  Add extra initialization here
	m_fileNameList.DeleteAllItems();
	m_fileNameList.SetExtendedStyle(LVS_EX_GRIDLINES);

	m_fileNameList.InsertColumn(0, _T("File Name"), LVCFMT_LEFT, 850, -1);

	return TRUE;  // return TRUE unless you set the focus to a control
				  // EXCEPTION: OCX Property Pages should return FALSE
}
