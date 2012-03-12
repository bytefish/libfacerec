#include "test_precomp.hpp"
#include "opencv2/ts/ts.hpp"

using namespace cv;
using namespace std;

#define CHECK_C

Size sz(200, 500);

class Core_PCATest : public cvtest::BaseTest
{
public:
    Core_PCATest() {}
protected:
    void run(int)
    {
        double diffPrjEps, diffBackPrjEps,
        prjEps, backPrjEps,
        evalEps, evecEps;
        int maxComponents = 100;
        Mat rPoints(sz, CV_32FC1), rTestPoints(sz, CV_32FC1);
        RNG& rng = ts->get_rng(); 
        
        rng.fill( rPoints, RNG::UNIFORM, Scalar::all(0.0), Scalar::all(1.0) );
        rng.fill( rTestPoints, RNG::UNIFORM, Scalar::all(0.0), Scalar::all(1.0) );
        
        PCA rPCA( rPoints, Mat(), CV_PCA_DATA_AS_ROW, maxComponents ), cPCA;
        
        // 1. check C++ PCA & ROW
        Mat rPrjTestPoints = rPCA.project( rTestPoints );
        Mat rBackPrjTestPoints = rPCA.backProject( rPrjTestPoints );
        
        Mat avg(1, sz.width, CV_32FC1 );
        reduce( rPoints, avg, 0, CV_REDUCE_AVG );
        Mat Q = rPoints - repeat( avg, rPoints.rows, 1 ), Qt = Q.t(), eval, evec;
        Q = Qt * Q;
        Q = Q /(float)rPoints.rows;
        
        eigen( Q, eval, evec );
        /*SVD svd(Q);
         evec = svd.vt;
         eval = svd.w;*/
        
        Mat subEval( maxComponents, 1, eval.type(), eval.data ),
        subEvec( maxComponents, evec.cols, evec.type(), evec.data );
        
    #ifdef CHECK_C
        Mat prjTestPoints, backPrjTestPoints, cPoints = rPoints.t(), cTestPoints = rTestPoints.t();
        CvMat _points, _testPoints, _avg, _eval, _evec, _prjTestPoints, _backPrjTestPoints;
    #endif
        
        // check eigen()
        double eigenEps = 1e-6;
        double err;
        for(int i = 0; i < Q.rows; i++ )
        {
            Mat v = evec.row(i).t();
            Mat Qv = Q * v;
            
            Mat lv = eval.at<float>(i,0) * v;
            err = norm( Qv, lv );
            if( err > eigenEps )
            {
                ts->printf( cvtest::TS::LOG, "bad accuracy of eigen(); err = %f\n", err );
                ts->set_failed_test_info( cvtest::TS::FAIL_BAD_ACCURACY );
                return;
            }
        }
        // check pca eigenvalues
        evalEps = 1e-6, evecEps = 1e-3;
        err = norm( rPCA.eigenvalues, subEval );
        if( err > evalEps )
        {
            ts->printf( cvtest::TS::LOG, "pca.eigenvalues is incorrect (CV_PCA_DATA_AS_ROW); err = %f\n", err );
            ts->set_failed_test_info( cvtest::TS::FAIL_BAD_ACCURACY );
            return;
        }
        // check pca eigenvectors
        for(int i = 0; i < subEvec.rows; i++)
        {
            Mat r0 = rPCA.eigenvectors.row(i);
            Mat r1 = subEvec.row(i);
            err = norm( r0, r1, CV_L2 );
            if( err > evecEps )
            {
                r1 *= -1;
                double err2 = norm(r0, r1, CV_L2);
                if( err2 > evecEps )
                {
                    Mat tmp;
                    absdiff(rPCA.eigenvectors, subEvec, tmp);
                    double mval = 0; Point mloc;
                    minMaxLoc(tmp, 0, &mval, 0, &mloc);
                    
                    ts->printf( cvtest::TS::LOG, "pca.eigenvectors is incorrect (CV_PCA_DATA_AS_ROW); err = %f\n", err );
                    ts->printf( cvtest::TS::LOG, "max diff is %g at (i=%d, j=%d) (%g vs %g)\n",
                               mval, mloc.y, mloc.x, rPCA.eigenvectors.at<float>(mloc.y, mloc.x),
                               subEvec.at<float>(mloc.y, mloc.x));
                    ts->set_failed_test_info( cvtest::TS::FAIL_BAD_ACCURACY );
                    return;
                }
            }
        }
        
        prjEps = 1.265, backPrjEps = 1.265;
        for( int i = 0; i < rTestPoints.rows; i++ )
        {
            // check pca project
            Mat subEvec_t = subEvec.t();
            Mat prj = rTestPoints.row(i) - avg; prj *= subEvec_t;
            err = norm(rPrjTestPoints.row(i), prj, CV_RELATIVE_L2);
            if( err > prjEps )
            {
                ts->printf( cvtest::TS::LOG, "bad accuracy of project() (CV_PCA_DATA_AS_ROW); err = %f\n", err );
                ts->set_failed_test_info( cvtest::TS::FAIL_BAD_ACCURACY );
                return;
            }
            // check pca backProject
            Mat backPrj = rPrjTestPoints.row(i) * subEvec + avg;
            err = norm( rBackPrjTestPoints.row(i), backPrj, CV_RELATIVE_L2 );
            if( err > backPrjEps )
            {
                ts->printf( cvtest::TS::LOG, "bad accuracy of backProject() (CV_PCA_DATA_AS_ROW); err = %f\n", err );
                ts->set_failed_test_info( cvtest::TS::FAIL_BAD_ACCURACY );
                return;
            }
        }
        
        // 2. check C++ PCA & COL
        cPCA( rPoints.t(), Mat(), CV_PCA_DATA_AS_COL, maxComponents );
        diffPrjEps = 1, diffBackPrjEps = 1;
        Mat ocvPrjTestPoints = cPCA.project(rTestPoints.t());
        err = norm(cv::abs(ocvPrjTestPoints), cv::abs(rPrjTestPoints.t()), CV_RELATIVE_L2 );
        if( err > diffPrjEps )
        {
            ts->printf( cvtest::TS::LOG, "bad accuracy of project() (CV_PCA_DATA_AS_COL); err = %f\n", err );
            ts->set_failed_test_info( cvtest::TS::FAIL_BAD_ACCURACY );
            return;
        }
        err = norm(cPCA.backProject(ocvPrjTestPoints), rBackPrjTestPoints.t(), CV_RELATIVE_L2 );
        if( err > diffBackPrjEps )
        {
            ts->printf( cvtest::TS::LOG, "bad accuracy of backProject() (CV_PCA_DATA_AS_COL); err = %f\n", err );
            ts->set_failed_test_info( cvtest::TS::FAIL_BAD_ACCURACY );
            return;
        }
        
    #ifdef CHECK_C
        // 3. check C PCA & ROW
        _points = rPoints;
        _testPoints = rTestPoints;
        _avg = avg;
        _eval = eval;
        _evec = evec;
        prjTestPoints.create(rTestPoints.rows, maxComponents, rTestPoints.type() );
        backPrjTestPoints.create(rPoints.size(), rPoints.type() );
        _prjTestPoints = prjTestPoints;
        _backPrjTestPoints = backPrjTestPoints;
        
        cvCalcPCA( &_points, &_avg, &_eval, &_evec, CV_PCA_DATA_AS_ROW );
        cvProjectPCA( &_testPoints, &_avg, &_evec, &_prjTestPoints );
        cvBackProjectPCA( &_prjTestPoints, &_avg, &_evec, &_backPrjTestPoints );
        
        err = norm(prjTestPoints, rPrjTestPoints, CV_RELATIVE_L2);
        if( err > diffPrjEps )
        {
            ts->printf( cvtest::TS::LOG, "bad accuracy of cvProjectPCA() (CV_PCA_DATA_AS_ROW); err = %f\n", err );
            ts->set_failed_test_info( cvtest::TS::FAIL_BAD_ACCURACY );
            return;
        }
        err = norm(backPrjTestPoints, rBackPrjTestPoints, CV_RELATIVE_L2);
        if( err > diffBackPrjEps )
        {
            ts->printf( cvtest::TS::LOG, "bad accuracy of cvBackProjectPCA() (CV_PCA_DATA_AS_ROW); err = %f\n", err );
            ts->set_failed_test_info( cvtest::TS::FAIL_BAD_ACCURACY );
            return;
        }
        
        // 3. check C PCA & COL
        _points = cPoints;
        _testPoints = cTestPoints;
        avg = avg.t(); _avg = avg;
        eval = eval.t(); _eval = eval;
        evec = evec.t(); _evec = evec;
        prjTestPoints = prjTestPoints.t(); _prjTestPoints = prjTestPoints;
        backPrjTestPoints = backPrjTestPoints.t(); _backPrjTestPoints = backPrjTestPoints;
        
        cvCalcPCA( &_points, &_avg, &_eval, &_evec, CV_PCA_DATA_AS_COL );
        cvProjectPCA( &_testPoints, &_avg, &_evec, &_prjTestPoints );
        cvBackProjectPCA( &_prjTestPoints, &_avg, &_evec, &_backPrjTestPoints );
        
        err = norm(cv::abs(prjTestPoints), cv::abs(rPrjTestPoints.t()), CV_RELATIVE_L2 );
        if( err > diffPrjEps )
        {
            ts->printf( cvtest::TS::LOG, "bad accuracy of cvProjectPCA() (CV_PCA_DATA_AS_COL); err = %f\n", err );
            ts->set_failed_test_info( cvtest::TS::FAIL_BAD_ACCURACY );
            return;
        }
        err = norm(backPrjTestPoints, rBackPrjTestPoints.t(), CV_RELATIVE_L2);
        if( err > diffBackPrjEps )
        {
            ts->printf( cvtest::TS::LOG, "bad accuracy of cvBackProjectPCA() (CV_PCA_DATA_AS_COL); err = %f\n", err );
            ts->set_failed_test_info( cvtest::TS::FAIL_BAD_ACCURACY );
            return;
        }
    #endif
    }
};

#include "helper.hpp"

class Core_HelperTest : public cvtest::BaseTest
{
public:
    Core_HelperTest() {}
    ~Core_HelperTest() {}

protected:
    void run(int a){

    }

    int checkSymmetry() {

    }
};

TEST(Core_PCA, regression) { Core_PCATest test; test.safe_run(); }
TEST(Core_Helper, regression) { Core_HelperTest test; test.safe_run(); }
