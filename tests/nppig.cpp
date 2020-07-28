////////////////////////////////////////////////////////////////////////////////
// I MAKE NO CLAIMS TO THIS FILE, @txbob (the great) KINDLY SUPPLIED THIS:    //
// https://devtalk.nvidia.com/default/topic/1037482/gpu-accelerated-libraries/help-me-help-you-with-modern-cmake-and-cuda-mwe-for-npp/post/5271071/#5271071
////////////////////////////////////////////////////////////////////////////////
#include <nppi_geometry_transforms.h>
#include <cuda_runtime_api.h>
#include <assert.h>
#include <iostream>
#include <gtest/gtest.h>
#define DIVUP(source,round) (((source+(round-1))/round)*round)

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

class flipTest : public testing::TestWithParam<int>
{
	public:
    int flip_code;

    virtual void SetUp()
    {
        flip_code = GetParam();
    }
};

TEST_P(flipTest, Accuracy)
{
    /**
     * 1 channel 8-bit unsigned image mirror.
     */
    const int simgrows = 113;
    const int simgcols = 113;
    Npp8u *d_pSrc, *d_pDst;
    NppiSize oROI;  oROI.width = simgcols;  oROI.height = simgrows;
    const int nSrcStep = DIVUP(simgcols  ,128)*sizeof(d_pSrc[0]);
    const int nDstStep = DIVUP(oROI.width,128)*sizeof(d_pDst[0]);
	//std::cout << "nSrcStep : " << nSrcStep << std::endl;
	//std::cout << "nDstStep : " << nDstStep << std::endl;
    const int simgsize = nSrcStep*   simgcols*sizeof(d_pSrc[0]);
    const int dimgsize = nDstStep*oROI.height*sizeof(d_pSrc[0]);
    //const int simgpix  = simgrows*simgcols;
    //const int dimgpix  = oROI.width*oROI.height;
    //const NppiAxis flip = NPP_VERTICAL_AXIS;
    Npp8u *h_img = new Npp8u[simgsize];
	//std::cout << "simgpix  : " << simgpix  << std::endl;
	//std::cout << "simgsize : " << simgsize << std::endl;
    for (int i = 0; i < simgrows; i++)
        for (int j = 0; j < simgcols; j++)
            h_img[i*nSrcStep+j] = simgcols-j-1;
    cudaError_t err = cudaMalloc((void **)&d_pSrc, simgsize);
    EXPECT_EQ(err, cudaSuccess);
    err = cudaMalloc((void **)&d_pDst, dimgsize);
    EXPECT_EQ(err, cudaSuccess);
    err = cudaMemcpy(d_pSrc, h_img, simgsize, cudaMemcpyHostToDevice);
    EXPECT_EQ(err, cudaSuccess);
	//std::cout << "LINE     : (" << __LINE__ << ")" << std::endl;
    err = cudaMemcpy(d_pDst, h_img, simgsize, cudaMemcpyHostToDevice);
    EXPECT_EQ(err, cudaSuccess);
    // perform mirror op
	//std::cout << "LINE     : (" << __LINE__ << ")" << std::endl;
	//std::cout << "flip_code: " << flip_code << std::endl;
    NppStatus ret = nppiMirror_8u_C1IR(d_pDst, nDstStep, oROI, 
					(flip_code == 0 ? NPP_HORIZONTAL_AXIS : (flip_code > 0 ? NPP_VERTICAL_AXIS : NPP_BOTH_AXIS)));
    EXPECT_EQ(ret, NPP_NO_ERROR);
    err = cudaMemcpy(h_img, d_pDst, dimgsize, cudaMemcpyDeviceToHost);
    EXPECT_EQ(err, cudaSuccess);
	err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess);
    // test for R to L flip
	std::string message;
	bool failedFlag = false;
    for (int i = 0; i < oROI.height; i++)
	{
        for (int j = 0; j < oROI.width; j++)
		{
            if(h_img[i*nDstStep+j] == j)
				message += 'o';
			else
			{
				failedFlag = true;
				message += 'x';
			}
		}
		message += '\n';
	}
	if(failedFlag)
	{
		std::cout << std::endl;
		std::cout << message;
	}
	EXPECT_FALSE(failedFlag);

	delete [] h_img;
	cudaFree((void*)&d_pSrc);
	cudaFree((void*)&d_pDst);
    //std::cout << "Test ran successfully!" << std::endl;
}

enum flipCode {FLIP_BOTH = 0, FLIP_X = 1, FLIP_Y = -1};
INSTANTIATE_TEST_CASE_P(nppiMirror, flipTest, testing::Values(0, 1, -1));
