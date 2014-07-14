//��ȥ����
int main()
{
	IplImage* pFrame = NULL; 
	IplImage* pFrImg = NULL;
	IplImage* pBkImg = NULL;

	CvMat* pFrameMat = NULL;
	CvMat* pFrMat = NULL;
	CvMat* pBkMat = NULL;

	CvCapture* pCapture = NULL;

	int nFrmNum = 0;

	//��������
	cvNamedWindow("video", 1);
	cvNamedWindow("background",1);
	cvNamedWindow("foreground",1);

	pCapture = cvCaptureFromFile("media.avi");
	while(pFrame = cvQueryFrame( pCapture ))
	{
		nFrmNum++;

		//����ǵ�һ֡����Ҫ�����ڴ棬����ʼ��
		if(nFrmNum == 1)
		{
			pBkImg = cvCreateImage(cvSize(pFrame->width, pFrame->height), IPL_DEPTH_8U, 1);
			pFrImg = cvCreateImage(cvSize(pFrame->width, pFrame->height), IPL_DEPTH_8U, 1);

			pBkMat = cvCreateMat(pFrame->height, pFrame->width, CV_32FC1);
			pFrMat = cvCreateMat(pFrame->height, pFrame->width, CV_32FC1);
			pFrameMat = cvCreateMat(pFrame->height, pFrame->width, CV_32FC1);

			//ת���ɵ�ͨ��ͼ���ٴ���
			cvCvtColor(pFrame, pBkImg, CV_BGR2GRAY);
			cvCvtColor(pFrame, pFrImg, CV_BGR2GRAY);

			cvConvert(pFrImg, pFrameMat);
			cvConvert(pFrImg, pFrMat);
			cvConvert(pFrImg, pBkMat);
		}
		else
		{
			cvCvtColor(pFrame, pFrImg, CV_BGR2GRAY);
			cvConvert(pFrImg, pFrameMat);
			//��ǰ֡������ͼ���
			cvAbsDiff(pFrameMat, pBkMat, pFrMat);
			//��ֵ��ǰ��ͼ
			cvThreshold(pFrMat, pFrImg, 60, 255.0, CV_THRESH_BINARY);
			//���±���
			cvRunningAvg(pFrameMat, pBkMat, 0.003, 0);
			//������ת��Ϊͼ���ʽ��������ʾ
			cvConvert(pBkMat, pBkImg);

			cvShowImage("video", pFrame);
			cvShowImage("background", pBkImg);
			cvShowImage("foreground", pFrImg);

			if( cvWaitKey(2) >= 0 )
				break;
		}
	}
	cvDestroyWindow("video");
	cvDestroyWindow("background");
	cvDestroyWindow("foreground");d
	cvReleaseImage(&pFrImg);
	cvReleaseImage(&pBkImg);
	cvReleaseMat(&pFrameMat);
	cvReleaseMat(&pFrMat);
	cvReleaseMat(&pBkMat);
	cvReleaseCapture(&pCapture);
	return 0;
}