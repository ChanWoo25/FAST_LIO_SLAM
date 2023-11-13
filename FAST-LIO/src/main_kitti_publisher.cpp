#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>
#include <fmt/format.h>
#include <fmt/printf.h>

#include <memory>
#include <chrono>
#include <thread>
using namespace std::chrono_literals;

#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <string>
#include <sstream>

struct Config {
  double voxel_size;
  /* kitti root folder */
  std::string root_dir;
} cfg;

Eigen::Matrix3d R_velo_wrt_imu; Eigen::Vector3d t_velo_wrt_imu;
Eigen::Matrix3d R_velo_wrt_cam; Eigen::Vector3d t_velo_wrt_cam;
Eigen::Matrix3d R0;

pcl::PointXYZI vec2point(const Eigen::Vector3d &vec) {
  pcl::PointXYZI pi;
  pi.x = vec[0];
  pi.y = vec[1];
  pi.z = vec[2];
  return pi;
}

Eigen::Vector3d point2vec(const pcl::PointXYZI &pi) {
  return Eigen::Vector3d(pi.x, pi.y, pi.z);
}

void addScan(
  const std::string & scan_fn,
  pcl::PointCloud<pcl::PointXYZI> & merge_cloud,
  const Eigen::Vector3d & translation,
  const Eigen::Matrix3d & rotation)
{
  std::ifstream f_bin;
  f_bin.open(scan_fn, std::ifstream::in | std::ifstream::binary);
  if (!f_bin.is_open())
  {
    std::cerr << "Fail to open " << scan_fn << std::endl;
  }

  f_bin.seekg(0, std::ios::end);
  const size_t num_elements = f_bin.tellg() / sizeof(float);
  std::vector<float> buf(num_elements);
  f_bin.seekg(0, std::ios::beg);
  f_bin.read(
    reinterpret_cast<char *>(
      &buf[0]),
      num_elements * sizeof(float));
  fmt::print("Add {} pts\n", num_elements/4);
  for (std::size_t i = 0; i < buf.size(); i += 4)
  {
    pcl::PointXYZI point;
    point.x = buf[i];
    point.y = buf[i + 1];
    point.z = buf[i + 2];
    point.intensity = buf[i + 3];

    Eigen::Vector3d pv = point2vec(point);
    pv = R_velo_wrt_cam * pv + t_velo_wrt_cam;
    pv = R0.transpose() * pv;
    pv = rotation * pv + translation;
    point = vec2point(pv);
    merge_cloud.push_back(point);
  }
}

void load_pose_with_time(
  const std::string &pose_file,
  std::vector<std::pair<Eigen::Vector3d, Eigen::Matrix3d>> &poses_vec,
  std::vector<double> &times_vec)
{
  times_vec.clear();
  poses_vec.clear();
  std::ifstream fin(pose_file);
  std::string line;
  Eigen::Matrix<double, 1, 7> temp_matrix;
  while (getline(fin, line)) {
    std::istringstream sin(line);
    std::vector<std::string> Waypoints;
    std::string info;
    int number = 0;
    while (getline(sin, info, ' ')) {
      if (number == 0) {
        double time;
        std::stringstream data;
        data << info;
        data >> time;
        times_vec.push_back(time);
        number++;
      } else {
        double p;
        std::stringstream data;
        data << info;
        data >> p;
        temp_matrix[number - 1] = p;
        if (number == 7) {
          Eigen::Vector3d translation(temp_matrix[0], temp_matrix[1],
                                      temp_matrix[2]);
          Eigen::Quaterniond q(temp_matrix[6], temp_matrix[3], temp_matrix[4],
                               temp_matrix[5]);
          std::pair<Eigen::Vector3d, Eigen::Matrix3d> single_pose;
          single_pose.first = translation;
          single_pose.second = q.toRotationMatrix();
          poses_vec.push_back(single_pose);
        }
        number++;
      }
    }
  }
}

void generateSubmap(
  const std::string & seq_dir,
  const double & init_voxel_size,
  const size_t & target_pc_size)
{
  std::string dat_fn = seq_dir + "/keyframes_dat_per5m_over10m.txt";
  std::string idx_fn = seq_dir + "/keyframes_idx_per5m_over10m.txt";
  std::string pos_fn = seq_dir + "/pose.txt";
  std::ifstream f_dat, f_idx;
  f_dat.open(dat_fn);
  f_idx.open(idx_fn);

  if (!f_idx.is_open()) { std::cerr << "Fail to open " << idx_fn << std::endl; }
  if (!f_dat.is_open()) { std::cerr << "Fail to open " << dat_fn << std::endl; }


  std::vector<std::pair<Eigen::Vector3d, Eigen::Matrix3d>> poses_vec;
  std::vector<double> times_vec;
  load_pose_with_time(pos_fn, poses_vec, times_vec);
  std::cout << "Sucessfully load pose with number: " << poses_vec.size()
            << std::endl;
  int prev_start {-1};
  while (f_idx.peek() != EOF)
  {
    int start, end;
    f_idx >> start >> end;
    if (prev_start == start) { break; }
    // std::cout << "start: " << start << ", end: " << end << std::endl;

    auto cloud = pcl::PointCloud<pcl::PointXYZI>().makeShared();
    for (int i = start; i <= end+1; i++)
    {
      auto scan_fn = fmt::format("{}/lidar/{:06d}.bin", seq_dir, i);
      // fmt::print("{}\n", scan_fn);
      addScan(scan_fn, (*cloud), poses_vec[i].first, poses_vec[i].second);
    }
    fmt::print("Merge num: {}\n", cloud->size());

    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZI> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.01);

    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::ExtractIndices<pcl::PointXYZI> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(inliers);
    extract.setNegative(false);
    extract.filter(*cloud_plane);

    pcl::visualization::PCLVisualizer viewer;
    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> color_handler(cloud, "intensity");
    viewer.setBackgroundColor (0, 0, 0);
    viewer.addCoordinateSystem (1.0);
    viewer.initCameraParameters ();
    viewer.setShowFPS(true);
    viewer.setCameraPosition(1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    viewer.addPointCloud<pcl::PointXYZI>(cloud, color_handler, "sample cloud");
    viewer.setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
      1, "sample cloud");
    while (!viewer.wasStopped())
    {
      viewer.spinOnce(100);
      std::this_thread::sleep_for(100ms);
    }
    break;
    prev_start = start;
  }
  f_dat.close();
  f_idx.close();
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "kitti_publisher");
  ros::NodeHandle nh;

  nh.param<double>("voxel_size", cfg.voxel_size, 1.0);
  nh.param<std::string>("root_dir", cfg.root_dir, "");
  ROS_ASSERT(!cfg.root_dir.empty());

  ros::Publisher pointCloudPublisher
    = nh.advertise<sensor_msgs::PointCloud2>(
        "/velodyne_points", 1);

  auto lidar_fn = fmt::format("{}/lidar_data.txt", cfg.root_dir);
  std::ifstream f_lidar(lidar_fn);
  std::vector<double> timestamps;
  std::vector<std::string> scan_fns;
  while (f_lidar.peek() != EOF)
  {
    std::string path;
    double time;
    f_lidar >> time >> path;
    if (path.empty()) { break; }
    timestamps.push_back(time);
    scan_fns.push_back(path);
  }

  for (size_t i = 0; i < timestamps.size(); i++)
  {

  }
  std::string dat_fn = seq_dir + "/keyframes_dat_per5m_over10m.txt";
  std::string idx_fn = seq_dir + "/keyframes_idx_per5m_over10m.txt";
  std::string pos_fn = seq_dir + "/pose.txt";


  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg()
  cloudMsg.header.stamp = ros::Time::now();
  cloudMsg.header.frame_id = "base_link";  // Set the appropriate frame ID

  std::vector<std::string> SEQS = {"00"}; // {"00", "02", "05", "08"};

  fmt::print("Sleep 2sec and start");
  std::this_thread::sleep_for(2s);


  for (const auto & seq: SEQS)
  {
    generateSubmap(cfg.root_dir, cfg.voxel_size, 4096UL);
  }

  fmt::print("End ... sleep 2sec and exit");
  std::this_thread::sleep_for(2s);

  return EXIT_SUCCESS;
}


//   /* For saving undistorted lidar points */
//   string save_root_dir = "/data/datasets/dataset_project";
//   nh.param<string>("seq_name", seq_name,"");
//   seq_dir = save_root_dir + "/" + seq_name;
//   seq_pcd_dir = seq_dir + "/lidar_undistorted";
//   lidar_fn = seq_dir + "/lidar_data.txt";
//   pose_fastlio2_fn = seq_dir + "/pose_fast_lio2.txt";
//   if (mkdir(seq_pcd_dir.c_str(), 777) == 0)
//   {
//     std::cout << "Directory created successfully." << std::endl;
//   }
//   else
//   {
//     std::cerr << "Error: Unable to create the directory." << std::endl;
//   }
//   // Open the file in append mode
//   f_lidar.open(lidar_fn, std::ios::app);
//   f_pose_fastlio2.open(pose_fastlio2_fn, std::ios::app);
//   if (!f_lidar.is_open()) { std::cerr << "Fail to open " << lidar_fn << std::endl; exit(1); }
//   if (!f_pose_fastlio2.is_open()) { std::cerr << "Fail to open " << pose_fastlio2_fn << std::endl; exit(1); }

//   cout<<"p_pre->lidar_type "<<p_pre->lidar_type<<endl;

//   path.header.stamp    = ros::Time::now();
//   path.header.frame_id ="camera_init";

//   /*** variables definition ***/
//   int effect_feat_num = 0, frame_num = 0;
//   double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
//   bool flg_EKF_converged, EKF_stop_flg = 0;

//   FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
//   HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);

//   _featsArray.reset(new PointCloudXYZI());

//   memset(point_selected_surf, true, sizeof(point_selected_surf));
//   memset(res_last, -1000.0f, sizeof(res_last));
//   downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
//   downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
//   memset(point_selected_surf, true, sizeof(point_selected_surf));
//   memset(res_last, -1000.0f, sizeof(res_last));

//   Lidar_T_wrt_IMU<<VEC_FROM_ARRAY(extrinT);
//   Lidar_R_wrt_IMU<<MAT_FROM_ARRAY(extrinR);
//   p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
//   p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
//   p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
//   p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
//   p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));

//   double epsi[23] = {0.001};
//   fill(epsi, epsi+23, 0.001);
//   kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);

//   /*** debug record ***/
//   FILE *fp;
//   string pos_log_dir = root_dir + "/Log/pos_log.txt";
//   fp = fopen(pos_log_dir.c_str(),"w");

//   ofstream fout_pre, fout_out, fout_dbg;
//   fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"),ios::out);
//   fout_out.open(DEBUG_FILE_DIR("mat_out.txt"),ios::out);
//   fout_dbg.open(DEBUG_FILE_DIR("dbg.txt"),ios::out);
//   if (fout_pre && fout_out)
//       cout << "~~~~"<<ROOT_DIR<<" file opened" << endl;
//   else
//       cout << "~~~~"<<ROOT_DIR<<" doesn't exist" << endl;

//   /*** ROS subscribe initialization ***/
//   ros::Subscriber sub_pcl = p_pre->lidar_type == AVIA ? \
//       nh.subscribe(lid_topic, 200000, livox_pcl_cbk) : \
//       nh.subscribe(lid_topic, 200000, standard_pcl_cbk);
//   ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);
//   ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>
//           ("/cloud_registered", 100000);
//   ros::Publisher pubLaserCloudFull_body = nh.advertise<sensor_msgs::PointCloud2>
//           ("/cloud_registered_body", 100000);
//   ros::Publisher pubLaserCloudFull_lidar = nh.advertise<sensor_msgs::PointCloud2>
//           ("/cloud_registered_lidar", 100000);
//   ros::Publisher pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>
//           ("/cloud_effected", 100000);
//   ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>
//           ("/Laser_map", 100000);
//   ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>
//           ("/Odometry", 100000);
//   ros::Publisher pubPath          = nh.advertise<nav_msgs::Path>
//           ("/path", 100000);
// //------------------------------------------------------------------------------------------------------
//   signal(SIGINT, SigHandle);
//   ros::Rate rate(5000);
//   bool status = ros::ok();
//   while (status)
//   {
//       if (flg_exit) break;
//       ros::spinOnce();
//       if(sync_packages(Measures))
//       {
//           if (flg_reset)
//           {
//               ROS_WARN("reset when rosbag play back");
//               p_imu->Reset();
//               flg_reset = false;
//               Measures.imu.clear();
//               continue;
//           }

//           double t0,t1,t2,t3,t4,t5,match_start, solve_start, svd_time;

//           match_time = 0;
//           kdtree_search_time = 0.0;
//           solve_time = 0;
//           solve_const_H_time = 0;
//           svd_time   = 0;
//           t0 = omp_get_wtime();

//           p_imu->Process(Measures, kf, feats_undistort); // point are undistorted.
//           state_point = kf.get_x();
//           pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

//           if (feats_undistort->empty() || (feats_undistort == NULL))
//           {
//               first_lidar_time = Measures.lidar_beg_time;
//               p_imu->first_lidar_time = first_lidar_time;
//               // cout<<"FAST-LIO not ready"<<endl;
//               continue;
//           }

//           flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? \
//                           false : true;
//           /*** Segment the map in lidar FOV ***/
//           lasermap_fov_segment();

//           /*** downsample the feature points in a scan ***/
//           downSizeFilterSurf.setInputCloud(feats_undistort);
//           downSizeFilterSurf.filter(*feats_down_body);
//           t1 = omp_get_wtime();
//           feats_down_size = feats_down_body->points.size();
//           /*** initialize the map kdtree ***/
//           if(ikdtree.Root_Node == nullptr)
//           {
//               if(feats_down_size > 5)
//               {
//                   ikdtree.set_downsample_param(filter_size_map_min);
//                   feats_down_world->resize(feats_down_size);
//                   for(int i = 0; i < feats_down_size; i++)
//                   {
//                       pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
//                   }
//                   ikdtree.Build(feats_down_world->points);
//               }
//               continue;
//           }
//           int featsFromMapNum = ikdtree.validnum();
//           kdtree_size_st = ikdtree.size();



//           // cout<<"[ mapping ]: In num: "<<feats_undistort->points.size()<<" downsamp "<<feats_down_size<<" Map num: "<<featsFromMapNum<<"effect num:"<<effct_feat_num<<endl;

//           /*** ICP and iterated Kalman filter update ***/
//           normvec->resize(feats_down_size);
//           feats_down_world->resize(feats_down_size);

//           V3D ext_euler = SO3ToEuler(state_point.offset_R_L_I);
//           state_point.rot.coeffs()[0];
//           fout_pre<<setw(20)<<Measures.lidar_beg_time - first_lidar_time<<" "<<euler_cur.transpose()<<" "<< state_point.pos.transpose()<<" "<<ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<< " " << state_point.vel.transpose() \
//           <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<< endl;

//           if(0) // If you need to see map point, change to "if(1)"
//           {
//               PointVector ().swap(ikdtree.PCL_Storage);
//               ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
//               featsFromMap->clear();
//               featsFromMap->points = ikdtree.PCL_Storage;
//           }

//           pointSearchInd_surf.resize(feats_down_size);
//           Nearest_Points.resize(feats_down_size);
//           int  rematch_num = 0;
//           bool nearest_search_en = true; //

//           t2 = omp_get_wtime();

//           /*** iterated state estimation ***/
//           double t_update_start = omp_get_wtime();
//           double solve_H_time = 0;
//           kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
//           state_point = kf.get_x();
//           euler_cur = SO3ToEuler(state_point.rot);
//           pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
//           geoQuat.x = state_point.rot.coeffs()[0];
//           geoQuat.y = state_point.rot.coeffs()[1];
//           geoQuat.z = state_point.rot.coeffs()[2];
//           geoQuat.w = state_point.rot.coeffs()[3];
//           double t_update_end = omp_get_wtime();

//           /* Log poses and undistorted points */
//           std::stringstream ss1;
//           ss1 << "lidar_undistorted/" << std::setfill('0') << std::setw(6) << scan_idx++ << ".bin";
//           std::string pcd_fn = ss1.str();
//           std::ofstream ofile;
//           ofile.open(std::string(seq_dir + "/" + pcd_fn), std::ios::out | std::ios::binary);
//           if (ofile.is_open())
//           {
//               for (const auto & point : feats_undistort->points)
//               {
//                   ofile.write(reinterpret_cast<const char *>(&point.x), sizeof(float));
//                   ofile.write(reinterpret_cast<const char *>(&point.y), sizeof(float));
//                   ofile.write(reinterpret_cast<const char *>(&point.z), sizeof(float));
//                   ofile.write(reinterpret_cast<const char *>(&point.intensity), sizeof(float));
//               }
//               ofile.close();
//               std::cout << "Write to " << seq_pcd_dir + "/" + pcd_fn << std::endl;
//               ofile.close();
//           } else {
//               std::cerr << "Fail write to " << seq_pcd_dir + "/" + pcd_fn << std::endl;
//           }

//           std::stringstream ss2; ss2.precision(8);
//           ss2 << std::fixed << Measures.lidar_beg_time << " " << pcd_fn << "\n";
//           f_lidar << ss2.str();
//           std::stringstream ss3;
//           ss3.precision(8);
//           ss3 << std::fixed << Measures.lidar_beg_time;
//           ss3.precision(6);
//           ss3 << std::fixed
//               << " " << state_point.pos(0)
//               << " " << state_point.pos(1)
//               << " " << state_point.pos(2)
//               << " " << state_point.rot.coeffs()[0]
//               << " " << state_point.rot.coeffs()[1]
//               << " " << state_point.rot.coeffs()[2]
//               << " " << state_point.rot.coeffs()[3] << "\n";
//           f_pose_fastlio2 << ss3.str();

//           /******* Publish odometry *******/
//           publish_odometry(pubOdomAftMapped);

//           /*** add the feature points to map kdtree ***/
//           t3 = omp_get_wtime();
//           map_incremental();
//           t5 = omp_get_wtime();

//           /******* Publish points *******/
//           publish_path(pubPath);
//           if (scan_pub_en || pcd_save_en)
//             publish_frame_world(pubLaserCloudFull);
//           if (scan_pub_en && scan_body_pub_en)
//           {
//             publish_frame_body(pubLaserCloudFull_body);
//             publish_frame_lidar(pubLaserCloudFull_lidar);
//           }
//           // publish_effect_world(pubLaserCloudEffect);
//           // publish_map(pubLaserCloudMap);



//           /*** Debug variables ***/
//           if (runtime_pos_log)
//           {
//               frame_num ++;
//               kdtree_size_end = ikdtree.size();
//               aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
//               aver_time_icp = aver_time_icp * (frame_num - 1)/frame_num + (t_update_end - t_update_start) / frame_num;
//               aver_time_match = aver_time_match * (frame_num - 1)/frame_num + (match_time)/frame_num;
//               aver_time_incre = aver_time_incre * (frame_num - 1)/frame_num + (kdtree_incremental_time)/frame_num;
//               aver_time_solve = aver_time_solve * (frame_num - 1)/frame_num + (solve_time + solve_H_time)/frame_num;
//               aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1)/frame_num + solve_time / frame_num;
//               T1[time_log_counter] = Measures.lidar_beg_time;
//               s_plot[time_log_counter] = t5 - t0;
//               s_plot2[time_log_counter] = feats_undistort->points.size();
//               s_plot3[time_log_counter] = kdtree_incremental_time;
//               s_plot4[time_log_counter] = kdtree_search_time;
//               s_plot5[time_log_counter] = kdtree_delete_counter;
//               s_plot6[time_log_counter] = kdtree_delete_time;
//               s_plot7[time_log_counter] = kdtree_size_st;
//               s_plot8[time_log_counter] = kdtree_size_end;
//               s_plot9[time_log_counter] = aver_time_consu;
//               s_plot10[time_log_counter] = add_point_size;
//               time_log_counter ++;
//               printf("[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f construct H: %0.6f \n",t1-t0,aver_time_match,aver_time_solve,t3-t1,t5-t3,aver_time_consu,aver_time_icp, aver_time_const_H_time);
//               ext_euler = SO3ToEuler(state_point.offset_R_L_I);
//               fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_point.pos.transpose()<< " " << ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<<" "<< state_point.vel.transpose() \
//               <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<<" "<<feats_undistort->points.size()<<endl;
//               dump_lio_state_to_log(fp);
//           }
//       }

//       status = ros::ok();
//       rate.sleep();
//   }

//   f_lidar.close();
//   f_pose_fastlio2.close();

//   /**************** save map ****************/
//   /* 1. make sure you have enough memories
//   /* 2. pcd save will largely influence the real-time performences **/
//   if (pcl_wait_save->size() > 0 && pcd_save_en)
//   {
//       string file_name = string("scans.pcd");
//       string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
//       pcl::PCDWriter pcd_writer;
//       cout << "current scan saved to /PCD/" << file_name<<endl;
//       pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
//   }

//   fout_out.close();
//   fout_pre.close();

//   if (runtime_pos_log)
//   {
//       vector<double> t, s_vec, s_vec2, s_vec3, s_vec4, s_vec5, s_vec6, s_vec7;
//       FILE *fp2;
//       string log_dir = root_dir + "/Log/fast_lio_time_log.csv";
//       fp2 = fopen(log_dir.c_str(),"w");
//       fprintf(fp2,"time_stamp, total time, scan point size, incremental time, search time, delete size, delete time, tree size st, tree size end, add point size, preprocess time\n");
//       for (int i = 0;i<time_log_counter; i++){
//           fprintf(fp2,"%0.8f,%0.8f,%d,%0.8f,%0.8f,%d,%0.8f,%d,%d,%d,%0.8f\n",T1[i],s_plot[i],int(s_plot2[i]),s_plot3[i],s_plot4[i],int(s_plot5[i]),s_plot6[i],int(s_plot7[i]),int(s_plot8[i]), int(s_plot10[i]), s_plot11[i]);
//           t.push_back(T1[i]);
//           s_vec.push_back(s_plot9[i]);
//           s_vec2.push_back(s_plot3[i] + s_plot6[i]);
//           s_vec3.push_back(s_plot4[i]);
//           s_vec5.push_back(s_plot[i]);
//       }
//       fclose(fp2);
//   }

//   return 0;
