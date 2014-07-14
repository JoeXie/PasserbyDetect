//减去背景
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

	//创建窗口
	cvNamedWindow("video", 1);
	cvNamedWindow("background",1);
	cvNamedWindow("foreground",1);

	pCapture = cvCaptureFromFile("media.avi");
	while(pFrame = cvQueryFrame( pCapture ))
	{
		nFrmNum++;

		//如果是第一帧，需要申请内存，并初始化
		if(nFrmNum == 1)
		{
			pBkImg = cvCreateImage(cvSize(pFrame->width, pFrame->height), IPL_DEPTH_8U, 1);
			pFrImg = cvCreateImage(cvSize(pFrame->width, pFrame->height), IPL_DEPTH_8U, 1);

			pBkMat = cvCreateMat(pFrame->height, pFrame->width, CV_32FC1);
			pFrMat = cvCreateMat(pFrame->height, pFrame->width, CV_32FC1);
			pFrameMat = cvCreateMat(pFrame->height, pFrame->width, CV_32FC1);

			//转化成单通道图像再处理
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
			//当前帧跟背景图相减
			cvAbsDiff(pFrameMat, pBkMat, pFrMat);
			//二值化前景图
			cvThreshold(pFrMat, pFrImg, 60, 255.0, CV_THRESH_BINARY);
			//更新背景
			cvRunningAvg(pFrameMat, pBkMat, 0.003, 0);
			//将背景转化为图像格式，用以显示
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