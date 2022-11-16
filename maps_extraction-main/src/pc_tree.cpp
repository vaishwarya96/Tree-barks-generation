#include <thread>         // std::thread
#include <mutex>          // std::mutex, std::unique_lock
#include <iostream>
#include <math.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_circle.h>
#include <tuple>
#include <map>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "StatVar.h"

using namespace cv;
typedef pcl::PointXYZRGB pc;
typedef std::tuple<int, int> my_tuple;

//Visualization
pcl::visualization::PCLVisualizer::Ptr rgbVis (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud)
{ 
    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D     Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
    viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "sample cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();

    return (viewer);
}

//Ransac
std::vector<int> get_ransac_inliers(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud)
{

    std::vector<int> inliers;
    pcl::SampleConsensusModelCircle2D<pcl::PointXYZRGB>::Ptr model (new pcl::SampleConsensusModelCircle2D<pcl::PointXYZRGB> (cloud));
    model->setRadiusLimits(0.0, 5.0);
    pcl::RandomSampleConsensus<pcl::PointXYZRGB> ransac (model);
    ransac.setDistanceThreshold(.01);
    ransac.computeModel();
    ransac.getInliers(inliers);

    Eigen::VectorXf coeff;
    ransac.getModelCoefficients (coeff);
    //std::cout<<"Coefficients are "<<coeff[0]<<" "<<coeff[1]<<" " <<coeff[2]<<std::endl;
    return(inliers);
}

//Circle least square fit
Eigen::VectorXf circle_ls(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud)
{
    float x_mean = 0.0;
    float y_mean = 0.0;
    float ui = 0.0;
    float vi = 0.0;
    float Suu = 0.0, Suv = 0.0, Svv = 0.0, Suuu = 0.0, Suvv = 0.0, Svuu = 0.0, Svvv = 0.0;
    float uc = 0.0, vc = 0.0, alpha = 0.0, xc = 0.0, yc= 0.0;
    Eigen::MatrixXf A(2,2);
    Eigen::MatrixXf B(2,1);

    for(size_t i=0; i<cloud->points.size(); i++)
    {
        x_mean += cloud->points[i].x;
        y_mean += cloud->points[i].z;
    }
    x_mean /= cloud->points.size();
    y_mean /= cloud->points.size();

    for(size_t i=0; i<cloud->points.size(); i++)
    {
        ui = cloud->points[i].x - x_mean;
        vi = cloud->points[i].z - y_mean;
        Suu += ui * ui;
        Suv += ui * vi;
        Svv += vi * vi;
        Suuu += ui * ui * ui;
        Suvv += ui * vi * vi;
        Svuu += vi * ui * ui;
        Svvv += vi * vi * vi;
    }

    A(0,0) = Suu;
    A(0,1) = Suv;
    A(1,0) = Suv;
    A(1,1) = Svv;
    B(0,0) = 0.5 * (Suuu + Suvv);
    B(1,0) = 0.5 * (Svvv + Svuu);

    Eigen::MatrixXf X = (A.transpose()*A).inverse()*(A.transpose()*B);

    uc = X(0);
    vc = X(1);

    xc = X(0) + x_mean;
    yc = X(1) + y_mean;
    alpha = uc*uc + vc*vc + (Suu + Svv)/cloud->points.size();

    float r = sqrt(alpha);
    //cout<<"Circle least square "<<xc<<" "<<yc<<" "<<r<<std::endl;
    Eigen::VectorXf center(3,1);
    center(0,0) = xc;
    center(1,0) = yc;
    center(2,0) = r;

    return center;
}

//Returns theta in degrees
float get_theta(float x, float y, Eigen::VectorXf centre)
{
    float deg = 0.0;
    float theta = atan2((y - centre[1]), (x - centre[0]));

    if(theta>0)
        deg = theta * 180/M_PI;
    else
        deg = (theta + 2*M_PI)*180/M_PI;

    return deg;
}

template <class value_type>
class GenMat {
    protected:
        size_t width_, height_;
        std::vector<value_type> data_;
    public:
        GenMat() : width_(0), height_(0) {}
        GenMat(size_t w, size_t h) {resize(w,h);}
        void resize(size_t w, size_t h) {
            width_ = w;
            height_ = h;
            data_.resize(w*h);
        }
        void clear() {
            data_.clear();
        }
        const value_type & operator()(size_t i, size_t j) const {
            assert(i < height_);
            assert(j < width_);
            return data_[i*width_ + j];
        }
        value_type & operator()(size_t i, size_t j) {
            assert(i < height_);
            assert(j < width_);
            return data_[i*width_ + j];
        }
        size_t width() const {return width_;}
        size_t height() const {return height_;}

        cv::Size size() const {return cv::Size(width_,height_);}
        
};

template <class C>
class ListMat : public GenMat< std::list<C> > {
    protected:
        typedef GenMat< std::list<C> > Parent;
    public:
        void clear_lists() {
            for (size_t i=0;i<Parent::data_.size();i++) {
                Parent::data_[i].clear();
            }
        }
};

class PixelStat {
    protected :
        StatVar r_,g_,b_;
        StatVarMedian radius_;
        Eigen::Vector2f tree_centre_;
        Eigen::Vector2f cell_centre_;
        static float sigma_rad, sigma_rgb;
    public:
        PixelStat() {}
        PixelStat(const Eigen::VectorXf & centre, float c_z, float c_theta) {
            tree_centre_ << centre[0],centre[1];
            cell_centre_ << c_z, c_theta;
        }

        void append(const std::list<pc> & l, bool reset = false) {
            std::list<pcl::PointXYZRGB>::const_iterator it;
            if (reset) {
                r_.reset();
                g_.reset();
                b_.reset();
                radius_.reset();
            }
            for(it = l.begin(); it!=l.end(); ++it) {
                const pcl::PointXYZRGB & p = *it;
                float X = p.x;
                float Y = p.y;
                float Z = p.z;
                int R = p.r;
                int G = p.g;
                int B = p.b;
                float theta = get_theta(X, Z, tree_centre_);

                float radius = hypot(tree_centre_[0]-X,tree_centre_[1]-Z);
                float dist = hypot(cell_centre_[0] - Y,remainder(cell_centre_[1] - theta,360));
                float weight_rad = exp(-0.5 * dist / (sigma_rad*sigma_rad));
                float weight_rgb = exp(-0.5 * dist / (sigma_rgb*sigma_rgb));    

                r_.add(R,weight_rgb);
                g_.add(G,weight_rgb);
                b_.add(B,weight_rgb);


                radius_.add(radius,weight_rad);
            }
        }

        void finalize() {
            r_.finalize();
            g_.finalize();
            b_.finalize();
            radius_.finalize();
        }

        bool empty() {
            return radius_.empty();
        }

        cv::Vec3b mean_rgb() const {
            assert(r_.ready()); assert(g_.ready()); assert(b_.ready());
            return cv::Vec3b(b_.mean,g_.mean,r_.mean);
        }
        cv::Vec3f std_rgb() const {
            assert(r_.ready()); assert(g_.ready()); assert(b_.ready());
            return cv::Vec3f(b_.std,g_.std,r_.std);
        }

        float mean_radius() const {
            assert(radius_.ready());
            return radius_.mean;
        }
        float median_radius() const {
            assert(radius_.ready());
            return radius_.median;
        }
        float std_radius() const {
            assert(radius_.ready());
            return radius_.std;
        }
};

float PixelStat::sigma_rad = 1.5;
float PixelStat::sigma_rgb = 0.5;

struct ThreadData {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
    float bottom_range;
    float z_step;
    float max_range;
    std::vector<Eigen::VectorXf> & centre_map;
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> layers;
    size_t g_index, t_index;
    std::mutex mtx;
    ThreadData( pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, float bottom_range, float z_step, float max_range, std::vector<Eigen::VectorXf> & centre_map) :
        cloud(cloud), bottom_range(bottom_range), z_step(z_step), max_range(max_range), centre_map(centre_map), g_index(0) {
        }

    void prepare() {
        mtx.lock();
        if (layers.empty()) {
            g_index = 0;
            layers.resize(ceil((max_range-bottom_range)/z_step));
            centre_map.resize(layers.size());
            for (size_t i=0;i<layers.size();i++) {
                layers[i] = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
            }
            for (size_t i=0; i<cloud->points.size(); i++) {
                pcl::PointXYZRGB P = cloud->points[i];
                int il = std::min<int>(layers.size()-1,std::max<int>(0,round((P.y-bottom_range)/z_step)));
                layers[il]->push_back(P);
            }
            t_index = 20;
        }
        mtx.unlock();
    }

    void process() {
        prepare(); 
        while (1) {
            int index = -1;
            mtx.lock();
            if (g_index < layers.size()) {
                index = g_index;
                g_index += 1;
                if ((100*index)/layers.size() > t_index) {
                    t_index += 20;
                    std::cout << t_index << "%" << std::endl;
                }
            }
            mtx.unlock();
            if (index < 0) return;

            centre_map[index] = circle_ls(layers[index]);
        }
    }
};

void thread_func(ThreadData * data) {
    data->process();
}

typedef std::vector< ListMat<pc> > PointListArray;

struct StatisticThread {
    const std::vector<float> & z_steps;
    const std::vector<float> & theta_steps;
    const std::vector<Eigen::VectorXf> & centre_map;
    const PointListArray & point_map;
    std::vector< GenMat<PixelStat> > & stat_map;
    unsigned int res;
    unsigned int batch_size;
    unsigned int next_index;
    std::mutex mtx;
    typedef std::pair<size_t, size_t> ICoord;
    std::vector<ICoord> todos;
    cv::Mat_<cv::Vec3b> texture;
    cv::Mat_<float> mean_radius;
    cv::Mat_<float> variance_map;

    StatisticThread( const std::vector<float> & z_steps, const std::vector<float> & theta_steps, const std::vector<Eigen::VectorXf> & centre_map,
            const PointListArray & point_map, std::vector< GenMat<PixelStat> > & stat_map, unsigned int res, unsigned int batch_size):
        z_steps(z_steps), theta_steps(theta_steps), centre_map(centre_map),
        point_map(point_map), stat_map(stat_map), res(res), batch_size(batch_size), next_index(0),
        texture(point_map[res].size(), cv::Vec3b(0,0,0)), mean_radius(point_map[res].size(), 0.0f), variance_map(point_map[res].size(), 0.0f)

    {
        for (size_t i=0;i<point_map[res].height();i++) {
            for (size_t j=0;j<point_map[res].width();j++) {
                todos.push_back(ICoord(i,j));
            }
        }
    }

    void process() {
        while (1) {
            unsigned int i_start, i_end;
            mtx.lock();
            i_start = next_index;
            if (next_index < todos.size()) {
                i_end = std::min<unsigned int>(todos.size(),i_start+batch_size);
                next_index += batch_size;
            }
            mtx.unlock();
            if (i_start >= todos.size()) {
                break;
            }
            for (unsigned int t=i_start;t<i_end;t++) {
                unsigned int k = res;
                unsigned int i = todos[t].first;
                unsigned int j = todos[t].second;
                float c_z = i * z_steps[k];
                float c_theta = j * theta_steps[k];
                PixelStat S(centre_map[c_z / z_steps[0]],c_z,c_theta);
                for (int l = -1;l <= +1; l++) {
                    if (int(i+l) < 0) continue;
                    if (int(i+l) >= int(point_map[k].height())) continue;
                    for (int m = -1; m <= +1; m++) {
                        S.append(point_map[k](i+l,(j+m)%point_map[k].width()));
                    }
                }
                S.finalize();
                if (S.empty()) {
                    for (size_t ktmp = k+1;ktmp < point_map.size();ktmp++) {
                        S = stat_map[ktmp](i/2,j/2);
                        if (!S.empty()) {
                            break;
                        }
                    }
                }
                assert(!S.empty());
                stat_map[k](i,j) = S;
                texture(i,j) = S.mean_rgb();
                mean_radius(i,j) = S.median_radius();
                // mean_radius(i,j) = S.mean_radius();
                variance_map(i,j) = S.std_radius();
              }
        }

    }
};
            
void thread_stats(StatisticThread * data) {
    data->process();
}


int main(int argc, char **argv)
{
    const unsigned int num_threads = 8;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);

    float d_min = 0.0;
    float d_max = 1.1;
    // float d_max = 1.15;

    //Load the Point Cloud
    if (pcl::io::loadPLYFile<pcl::PointXYZRGB> (argv[1], *cloud) == -1)
    {
        PCL_ERROR("Couldn't read file \n");
        return(-1);
    }

    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
    pcl::ExtractIndices<pcl::PointXYZRGB> extract;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pt (new pcl::PointCloud<pcl::PointXYZRGB>);

    //Copy cloud to pt
    copyPointCloud(*cloud, *pt);
    std::cout<<"Size of the point cloud is "<<cloud->points.size()<<std::endl;

    //Get the indices of the points from the top and bottom of the tree
    for (size_t i=0; i<cloud->points.size();i++)
    {
        if (pt->points[i].y <= d_min || pt->points[i].y >=d_max)
        {
            inliers->indices.push_back(i);

        }
    }

    //Filter out the points (filtering the points from cloud)
    extract.setInputCloud(cloud);
    extract.setIndices(inliers);
    extract.setNegative(true);
    extract.filter(*cloud);
    std::cout<<"Size now is "<<cloud->points.size()<<std::endl;
    /*
    //Visualize the filtered point cloud
    pcl::visualization::PCLVisualizer::Ptr viewer;

    viewer = rgbVis(cloud);
    while(!viewer->wasStopped())
    {
    viewer->spinOnce(100);
    //this_thread::sleep_for(100ms);
    }
    */


    //Copy cloud to new_cloud
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr new_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);




    //You can remove the next 2 lines in the future
    copyPointCloud(*cloud, *new_cloud);
    pcl::PointIndices::Ptr layer_inliers(new pcl::PointIndices());




    //Divide the point cloud into layers based on the height
    float z_step = 0.001;
    float theta_step = 0.5;
    //int flag = 0;


    std::vector<Eigen::VectorXf> centre_map;
    ofstream cen("centre.txt");
    std::cout << "Extracting circle parameters" << std::endl;
#if 0
    for (float bottom_range = d_min; bottom_range < d_max; bottom_range += z_step) {
        //Copy cloud to new_cloud
        // copyPointCloud(*cloud, *new_cloud);
        pcl::PointIndices::Ptr layer_inliers(new pcl::PointIndices());
        //Get the points which are present in the given layer
        float top_range = bottom_range + z_step;
        std::cout << bottom_range << " -> " << top_range << std::endl;
        for (size_t i=0; i<cloud->points.size(); i++)
        {
            if (cloud->points[i].y <= bottom_range || cloud->points[i].y >= top_range)
            {
                layer_inliers->indices.push_back(i);
            }

        }

        //Filter out all other points
        extract.setInputCloud(cloud);
        extract.setIndices(layer_inliers);
        extract.setNegative(true);
        extract.filter(*new_cloud);


        //Get the inliers from RANSAC
        //std::vector<int> ransac_in = get_ransac_inliers(new_cloud);
        //pcl::PointCloud<pcl::PointXYZRGB>::Ptr final (new pcl::PointCloud<pcl::PointXYZRGB>);

        //final has the inlier points. new_cloud has the points in the layer.
        //copyPointCloud(*new_cloud, ransac_in, *final);

        //Get the centre of the circle from the inlier points

        Eigen::VectorXf centre = circle_ls(new_cloud);
        centre_map.push_back(centre);
        cen<<std::fixed<<setprecision(8)<<centre[1]<<" "<<centre[2]<<std::endl;
    } 
#else 
    ThreadData td(cloud,d_min,z_step,d_max,centre_map);
    std::thread th[num_threads];
    for (size_t i=0;i<num_threads;i++) {
        th[i] = std::thread(thread_func,&td);
    }
    for (size_t i=0;i<num_threads;i++) {
        th[i].join();
    }
    for (size_t i=0;i<centre_map.size();i++) {
        cen<<std::fixed<<setprecision(8)<<centre_map[i][0]<<" "<<centre_map[i][1]<<std::endl;
    }
#endif
    cen.close();
    std::cout << "Finished computing circle centerssssssssss. Creating data structure" << std::endl;


    size_t num_res = 8;
    PointListArray point_map(num_res);
    std::vector< GenMat<PixelStat> > stat_map(num_res);
    std::vector<float> divider(num_res,1);
    std::vector<float> z_steps(num_res,z_step);
    std::vector<float> theta_steps(num_res,theta_step);
    for (size_t i=1;i<num_res;i++) {
        divider[i] = 2*divider[i-1];
        z_steps[i] = 2*z_steps[i-1];
        theta_steps[i] = 2*theta_steps[i-1];
    }
    std::vector<size_t> num_z(num_res);
    std::vector<size_t> num_theta(num_res);
    for (size_t i=0;i<num_res;i++) {
        num_z[i] = ceil(float(centre_map.size())/divider[i]);
        num_theta[i] = ceil(360. / theta_steps[i]);
        point_map[i].resize(num_theta[i],num_z[i]);
        stat_map[i].resize(num_theta[i],num_z[i]);
    }

    std::cout << "Filling data structure" << std::endl;
    for (size_t i=0;i<cloud->size();i++) {
        const pcl::PointXYZRGB & p = cloud->points[i];
        int i_z = std::min<int>(centre_map.size()-1,std::max<int>(0,int(round((p.y - d_min)/z_step))));
        assert(i_z>=0); assert(i_z < int(centre_map.size()));
        float deg = get_theta(cloud->points[i].x, new_cloud->points[i].z, centre_map[i_z]);
        for (size_t k=0;k<num_res;k++) {
            int k_z = std::min<int>(point_map[k].height()-1,std::max<int>(0,round((p.y - d_min)/z_steps[k])));
            int i_theta = std::min<int>(point_map[k].width()-1,std::max<int>(0,round(deg/theta_steps[k])));
            assert(k < point_map.size());
            assert(k_z>=0);assert(k_z < int(point_map[k].height()));
            assert(i_theta>=0);assert(i_theta < int(point_map[k].width()));
            point_map[k](k_z,i_theta).push_back(p);
        }
    }
    std::cout << "Building statistics" << std::endl;

    unsigned int batch_size = point_map[num_res-1].width()*point_map[num_res-1].height() / num_threads / 2;
    for (size_t k_=0;k_<point_map.size();k_++) {
        // going in reverse order, coarser resolution first
        size_t k = point_map.size()-k_-1;
#if 0
        cv::Mat_<cv::Vec3b> texture(point_map[k].size(), cv::Vec3b(0,0,0));
        cv::Mat_<float> mean_radius(point_map[k].size(), 0.0f);
        cv::Mat_<float> variance_map(point_map[k].size(), 0.0f);
        for (size_t i=0;i<point_map[k].height();i++) {
            for (size_t j=0;j<point_map[k].width();j++) {
                float c_z = i * z_steps[k];
                float c_theta = j * theta_steps[k];
                PixelStat S(centre_map[c_z / z_step],c_z,c_theta);
                for (int l = -1;l <= +1; l++) {
                    if (int(i+l) < 0) continue;
                    if (int(i+l) >= int(point_map[k].height())) continue;
                    for (int m = -1; m <= +1; m++) {
                        S.append(point_map[k](i+l,(j+m)%point_map[k].width()));
                    }
                }
                S.finalize();
                if (S.empty()) {
                    for (size_t ktmp = k+1;ktmp < point_map.size();ktmp++) {
                        S = stat_map[ktmp](i/2,j/2);
                        if (!S.empty()) {
                            break;
                        }
                    }
                }
                assert(!S.empty());
                stat_map[k](i,j) = S;
                texture(i,j) = S.mean_rgb();
                mean_radius(i,j) = S.median_radius();
                // mean_radius(i,j) = S.mean_radius();
                variance_map(i,j) = S.std_radius();
            }
        }
        char fname1[1024],fname2[1024];
        sprintf(fname1,"texture%d.png",int(k));
        sprintf(fname2,"mean_radius%d.png",int(k));
        cv::imwrite(fname1, texture);
        normalize(mean_radius, mean_radius, 0 ,255, NORM_MINMAX);
        cv::imwrite(fname2, mean_radius);
#else
        batch_size *= 2;
        std::cout << "Res map " << k << std::endl;
        StatisticThread st(z_steps, theta_steps, centre_map,
                point_map, stat_map, k, batch_size);
        std::thread th[num_threads];
        for (size_t i=0;i<num_threads;i++) {
            th[i] = std::thread(thread_stats,&st);
        }
        for (size_t i=0;i<num_threads;i++) {
            th[i].join();
        }
        char fname1[1024],fname2[1024];
        sprintf(fname1,"texture%d.png",int(k));
        sprintf(fname2,"mean_radius%d.png",int(k));
        cv::imwrite(fname1, st.texture);
        ofstream rad("rad.txt");
        for (size_t i=0;i<st.mean_radius.rows;i++){
          for (size_t j=0;j<st.mean_radius.cols;j++){
            rad<<std::fixed<<setprecision(8)<<st.mean_radius[i][j]<<std::endl;
          }
        }
        rad.close();
        normalize(st.mean_radius, st.mean_radius, 0 ,255, NORM_MINMAX);
        cv::imwrite(fname2, st.mean_radius);
#endif 
    }
    std::cout << "Done" << std::endl;


    return 0;


}



