// CloudCropper parseable CLI for G3Reg global registration.
// Usage: cc_g3reg_cli <config> <src.pcd> <tgt.pcd>
//   <config> may be absolute or relative to the G3Reg project dir.
// glog is redirected to stderr; stdout carries ONLY the three parseable lines:
//   G3REG_TF: m00 m01 ... m33    (row-major 4x4, target<-source)
//   G3REG_INLIERS: <plane+line+cluster>
//   G3REG_TIME: <seconds>
#include "utils/config.h"
#include <iomanip>
#include <iostream>
#include <string>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "back_end/reglib.h"
#include "global_definition/global_definition.h"
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <glog/logging.h>

using namespace std;
using namespace clique_solver;
using namespace g3reg;

int main(int argc, char **argv) {
    if (argc < 4) {
        std::cerr << "Usage: cc_g3reg_cli <config> <src.pcd> <tgt.pcd>" << std::endl;
        return 2;
    }
    FLAGS_logtostderr = 1;  // keep stdout clean for the parser

    std::string cfg = argv[1];
    if (!cfg.empty() && cfg[0] != '/') cfg = config.project_path + "/" + cfg;
    InitGLOG(cfg, argv);
    config.load_config(cfg, argv);

    pcl::PointCloud<pcl::PointXYZ>::Ptr source(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr target(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(std::string(argv[2]), *source) == -1) {
        std::cerr << "ERROR: cannot read source pcd: " << argv[2] << std::endl;
        return 1;
    }
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(std::string(argv[3]), *target) == -1) {
        std::cerr << "ERROR: cannot read target pcd: " << argv[3] << std::endl;
        return 1;
    }

    FRGresult sol = g3reg::GlobalRegistration(source, target);
    const Eigen::Matrix4d &tf = sol.tf;

    std::cout << "G3REG_TF:";
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            std::cout << " " << std::setprecision(9) << tf(i, j);
    std::cout << "\n";
    std::cout << "G3REG_INLIERS: "
              << (sol.plane_inliers + sol.line_inliers + sol.cluster_inliers) << "\n";
    std::cout << "G3REG_TIME: " << sol.total_time << "\n";
    std::cout.flush();
    return 0;
}
