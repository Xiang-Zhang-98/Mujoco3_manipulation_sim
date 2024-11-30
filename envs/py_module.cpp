#include <iostream>
#include <stdlib.h>
#include "stdio.h"
#include "string.h"

#include "mujoco.h"
#include "glfw3.h"
#include "mjxmacro.h"
#include "uitools.h"

using namespace std;

//-------------------------------- global -----------------------------------------------
// MuJoCo data structures
mjModel *m = NULL;         // MuJoCo model
mjData *d = NULL;          // MuJoCo data
GLFWwindow *window = NULL; // MuJoCo window
mjvCamera cam;             // abstract camera
mjvOption opt;             // visualization options
mjvScene scn;              // abstract scene
mjrContext con;            // custom GPU context
mjtNum timer_render = 0;   // init timer for control and ui
int controller = 0;        // controller index: 0: computed torque (track traj & vel); 1: impedance control; 2: eef apply force

// controller parameters
mjtNum ref_joint[8] = {0, 0, 0, 0, 0, 0, 0, 0};
mjtNum ref_vel[8] = {0, 0, 0, 0, 0, 0, 0, 0};
mjtNum ref_acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
mjtNum vel_ctl_integral[8] = {0, 0, 0, 0, 0, 0, 0, 0};
mjtNum Full_jacobian_global[48];
mjtNum kp[64];
mjtNum kv[64];
mjtNum ki[64];

#define PI 3.1415926535

//print variables in the control loop
bool Verbose = true;
bool Render = true;

// gripper status: 0.042 gripper fully close; 0.0 fully open
double gripper_pose = 0.0;
        
// mouse interaction
bool button_left = false;
bool button_middle = false;
bool button_right = false;
double lastx = 0;
double lasty = 0;

// constants
const int maxgeom = 5000;         // preallocated geom array in mjvScene
const double syncmisalign = 0.1;  // maximum time mis-alignment before re-sync
const double refreshfactor = 0.7; // fraction of refresh available for simulation

// keyboard callback
void keyboard(GLFWwindow *window, int key, int scancode, int act, int mods)
{
    // backspace: reset simulation
    if (act == GLFW_PRESS && key == GLFW_KEY_BACKSPACE)
    {
        mj_resetData(m, d);
        mj_forward(m, d);
    }
}

// mouse button callback
void mouse_button(GLFWwindow *window, int button, int act, int mods)
{
    // update button state
    button_left = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
    button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);
    button_right = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);

    // update mouse position
    glfwGetCursorPos(window, &lastx, &lasty);
}

// mouse move callback
void mouse_move(GLFWwindow *window, double xpos, double ypos)
{
    // no buttons down: nothing to do
    if (!button_left && !button_middle && !button_right)
        return;

    // compute mouse displacement, save
    double dx = xpos - lastx;
    double dy = ypos - lasty;
    lastx = xpos;
    lasty = ypos;

    // get current window size
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    // get shift key state
    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                      glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);

    // determine action based on mouse button
    mjtMouse action;
    if (button_right)
        action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    else if (button_left)
        action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    else
        action = mjMOUSE_ZOOM;

    // move camera
    mjv_moveCamera(m, action, dx / height, dy / height, &scn, &cam);
}

// scroll callback
void scroll(GLFWwindow *window, double xoffset, double yoffset)
{
    // emulate vertical mouse motion = 5% of window height
    mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05 * yoffset, &scn, &cam);
}

void update_ui(GLFWwindow *window)
{
    // get framebuffer viewport
    mjrRect viewport = {0, 0, 0, 0};
    glfwGetFramebufferSize(window, &viewport.width, &viewport.height);
    // update scene and render
    mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
    mjr_render(viewport, &scn, &con);
    // swap OpenGL buffers (blocking call due to v-sync)
    glfwSwapBuffers(window);
    // process pending GUI events, call GLFW callbacks
    glfwPollEvents();
    timer_render = d->time;
}

void update_controller_type(int controller_idx)
{
    controller = int(controller_idx);
}

void Euler2quat(mjtNum *Euler, mjtNum *quat)
{
    mjtNum roll = Euler[0];
    mjtNum pitch = Euler[1];
    mjtNum yaw = Euler[2];
    quat[1] = sin(roll / 2.0) * cos(pitch / 2.0) * cos(yaw / 2.0) - cos(roll / 2.0) * sin(pitch / 2.0) * sin(yaw / 2.0);
    quat[2] = cos(roll / 2.0) * sin(pitch / 2.0) * cos(yaw / 2.0) + sin(roll / 2.0) * cos(pitch / 2.0) * sin(yaw / 2.0);
    quat[3] = cos(roll / 2.0) * cos(pitch / 2.0) * sin(yaw / 2.0) + sin(roll / 2.0) * sin(pitch / 2.0) * cos(yaw / 2.0);
    quat[0] = cos(roll / 2.0) * cos(pitch / 2.0) * cos(yaw / 2.0) + sin(roll / 2.0) * sin(pitch / 2.0) * sin(yaw / 2.0);
}

void extract_submatrix(mjtNum* input, mjtNum* output, int start_row, int end_row, int start_col, int end_col)
{
    for (int i = start_row; i < end_row; i++)
    {
        for (int j = start_col; j < end_col; j++)
        {
            output[8 * i + j] = input[m->nv * i + j];
        }
    }
}

void eef_full_jacobian(mjtNum *Full_jacobian)
{
    //compute the full jacobian of the end effector(link 6)
    mjtNum jacp[3*m->nv];
    mjtNum jacr[3*m->nv];
    int geom_id=7;
    mj_jacGeom(m, d, jacp, jacr, int(geom_id));
    for (int j = 0; j < 3*m->nv; j++)
        Full_jacobian[j] = jacp[j];
    for (int j = 0; j < 3*m->nv; j++)
        Full_jacobian[j + 3*m->nv] = jacr[j];/**/
}

void computed_torque_control(const mjModel *m, mjData *d)
{
    // d->ctrl controls the actuator in xml file; d->qfrc_applied torques specified in joint space
    // If in xml, actuators are position/velocity, then d->ctrl here means reference position/velocity
    // d->ctrl 0-5 are joints torque's control signal; 6,7 are gripper position's control signal
    // d->qpos, d->qvel, d->qacc are joint positions, velocities, and acculations (0-5 for arm joints; 6,7 for gripper)
    // m->nq: number of generalized coordinates = dim(qpos)
    // m->nv: number of degrees of freedom = dim(qvel)
    // m->nu: number of actuators/controls = dim(ctrl)

    // ctrl = M * (ref_acc - kv*e_dot - ki*sum(e_dot)) + qfrc_inverse

    mj_rne(m, d, 0, d->qfrc_inverse);
    for (int i = 0; i < m->nv; i++)
        //d->qfrc_inverse[i] += - d->qfrc_passive[i] - d->qfrc_constraint[i];
        d->qfrc_inverse[i] += m->dof_armature[i] * d->qacc[i] - d->qfrc_passive[i] - d->qfrc_constraint[i];
    mjtNum e[8];               // theta - theta_d
    mjtNum e_dot[8];           // theta_dot - theta_d_dot
    mjtNum kve_dot[8];         // kv * e_dot
    mjtNum kpe[8];             // kp * e
    mjtNum inertial_pd[8];     // ref_acc - kv*e_dot - kp*e
    mjtNum inertial_torque[8]; // M * (ref_acc - kv*e_dot - kp*e)
    mju_sub(e, d->qpos, ref_joint, 8);
    mju_sub(e_dot, d->qvel, ref_vel, 8);
    mju_mulMatVec(kve_dot, kv, e_dot, 8, 8);
    mju_mulMatVec(kpe, kp, e, 8, 8);
    mju_sub(inertial_pd, ref_acc, kve_dot, 8);
    mju_subFrom(inertial_pd, kpe, 8);

    // mj_mulM(m, d, inertial_torque, inertial_pd); // this no longer works when there are objects in the scene
    mjtNum M[m->nv*m->nv];
    mjtNum M_robot[m->nu*m->nu];
    mj_fullM(m, M, d->qM);
    extract_submatrix(M, M_robot, 0, m->nu, 0, m->nu);
    mju_mulMatVec(inertial_torque, M_robot, inertial_pd, 8, 8);

    mju_add(d->ctrl, inertial_torque, d->qfrc_inverse, 8);
    // set gripper ctrl
    d->ctrl[6] = gripper_pose;
    d->ctrl[7] = gripper_pose;
}

void joint_vel_control(const mjModel *m, mjData *d)
{
    // d->ctrl controls the actuator in xml file; d->qfrc_applied torques specified in joint space
    // If in xml, actuators are position/velocity, then d->ctrl here means reference position/velocity
    // d->ctrl 0-5 are joints torque's control signal; 6,7 are gripper position's control signal
    // d->qpos, d->qvel, d->qacc are joint positions, velocities, and acculations (0-5 for arm joints; 6,7 for gripper)
    // m->nq: number of generalized coordinates = dim(qpos)
    // m->nv: number of degrees of freedom = dim(qvel)
    // m->nu: number of actuators/controls = dim(ctrl)

    // ctrl = M * (ref_acc - kv*e_dot - kp*e) + qfrc_inverse
    mj_rne(m, d, 0, d->qfrc_inverse);
    for (int i = 0; i < m->nv; i++)
        //d->qfrc_inverse[i] += - d->qfrc_passive[i] - d->qfrc_constraint[i];
        d->qfrc_inverse[i] += m->dof_armature[i] * d->qacc[i] - d->qfrc_passive[i] - d->qfrc_constraint[i];
    mjtNum e_dot[8];           // theta_dot - theta_d_dot
    mjtNum kpe_dot[8];         // kv * e_dot
    mjtNum ki_e_int[8];             // kp * e
    mjtNum inertial_pd[8];     // ref_acc - kv*e_dot - kp*e
    mjtNum inertial_torque[8]; // M * (ref_acc - kv*e_dot - kp*e)
    /*mju_sub(e, d->qpos, ref_joint, 8);*/
    mju_sub(e_dot, d->qvel, ref_vel, 8);
    mju_addTo(vel_ctl_integral, e_dot, 8);
    mju_mulMatVec(kpe_dot, kp, e_dot, 8, 8);
    mju_mulMatVec(ki_e_int, ki, vel_ctl_integral, 8, 8);
    mju_sub(inertial_pd, ref_acc, kpe_dot, 8);
    mju_subFrom(inertial_pd, ki_e_int, 8);

    // mj_mulM(m, d, inertial_torque, inertial_pd); // this no longer works when there are objects in the scene
    mjtNum M[m->nv*m->nv];
    mjtNum M_robot[m->nu*m->nu];
    mj_fullM(m, M, d->qM);
    extract_submatrix(M, M_robot, 0, m->nu, 0, m->nu);
    mju_mulMatVec(inertial_torque, M_robot, inertial_pd, 8, 8);

    mju_add(d->ctrl, inertial_torque, d->qfrc_inverse, 8);
    // set gripper ctrl
    d->ctrl[6] = gripper_pose;
    d->ctrl[7] = gripper_pose;
}

void cartesian_vel_control(const mjModel *m, mjData *d)
{
    // d->ctrl controls the actuator in xml file; d->qfrc_applied torques specified in joint space
    // If in xml, actuators are position/velocity, then d->ctrl here means reference position/velocity
    // d->ctrl 0-5 are joints torque's control signal; 6,7 are gripper position's control signal
    // d->qpos, d->qvel, d->qacc are joint positions, velocities, and acculations (0-5 for arm joints; 6,7 for gripper)
    // m->nq: number of generalized coordinates = dim(qpos)
    // m->nv: number of degrees of freedom = dim(qvel)
    // m->nu: number of actuators/controls = dim(ctrl)

    // ctrl = M * (ref_acc - kv*e_dot - kp*e) + qfrc_inverse

//    mjtNum Full_jacobian[48];
//    eef_full_jacobian(Full_jacobian);
    

    mj_rne(m, d, 0, d->qfrc_inverse);
    for (int i = 0; i < m->nv; i++)
        //d->qfrc_inverse[i] += - d->qfrc_passive[i] - d->qfrc_constraint[i];
        d->qfrc_inverse[i] += m->dof_armature[i] * d->qacc[i] - d->qfrc_passive[i] - d->qfrc_constraint[i];
    mjtNum e_dot[8];           // theta_dot - theta_d_dot
    mjtNum kpe_dot[8];         // kv * e_dot
    mjtNum ki_e_int[8];             // kp * e
    mjtNum inertial_pd[8];     // ref_acc - kv*e_dot - kp*e
    mjtNum inertial_torque[8]; // M * (ref_acc - kv*e_dot - kp*e)
    /*mju_sub(e, d->qpos, ref_joint, 8);*/
    mju_sub(e_dot, d->qvel, ref_vel, 8);
    mju_addTo(vel_ctl_integral, e_dot, 8);
    mju_mulMatVec(kpe_dot, kp, e_dot, 8, 8);
    mju_mulMatVec(ki_e_int, ki, vel_ctl_integral, 8, 8);
    mju_sub(inertial_pd, ref_acc, kpe_dot, 8);
    mju_subFrom(inertial_pd, ki_e_int, 8);

    // mj_mulM(m, d, inertial_torque, inertial_pd); // this no longer works when there are objects in the scene
    mjtNum M[m->nv*m->nv];
    mjtNum M_robot[m->nu*m->nu];
    mj_fullM(m, M, d->qM);
    extract_submatrix(M, M_robot, 0, m->nu, 0, m->nu);
    mju_mulMatVec(inertial_torque, M_robot, inertial_pd, 8, 8);

    mju_add(d->ctrl, inertial_torque, d->qfrc_inverse, 8);
    // set gripper ctrl
    d->ctrl[6] = gripper_pose;
    d->ctrl[7] = gripper_pose;
}

void cartesian_impedance_control(const mjModel *m, mjData *d)
{
    //Now it's a simplified Impedance Controller
//    cout << "nu:" << m->nu << endl;
//    cout << "nv:" << m->nv << endl;
    if (1)
    {
        //set ref_link6_pos
        mjtNum ref_link6_pos[3] = {-0.0, -0.5, 0.5};
        mjtNum ref_link6_rot_euler[3] = {0, PI / 2.0, 0};
        mjtNum ref_link6_rot[4];
        Euler2quat(ref_link6_rot_euler, ref_link6_rot);

        //calculate qfrc_inverse
        mj_rne(m, d, 0, d->qfrc_inverse);
        for (int i = 0; i < m->nv; i++)
            d->qfrc_inverse[i] += m->dof_armature[i] * d->qacc[i] - d->qfrc_passive[i] - d->qfrc_constraint[i];

        mjtNum inertial_torque[8];
        mjtNum force[6];
        mjtNum force_torque[m->nv]; //torque transformed from force
        mjtNum jacp[3*m->nv];
        mjtNum jacr[3*m->nv];
        mjtNum Full_jacobian[6*m->nv];
        mjtNum link6_pos[3];
        mjtNum link6_rot_mat[9];
        mjtNum link6_rot_quat[4];
        mjtNum ref_link6_vel[6];
        mjtNum e[6];
        //set feedback gains
        mjtNum kp[6] = {3.0, 3.0, 3.0, 9.0, 9.0, 9.0};
        mjtNum kd[6] = {12.0, 12.0, 12.0, 9.0, 9.0, 9.0};

        //get current link6_pos
        for (int j = 0; j < 3; j++)
            link6_pos[j] = d->geom_xpos[3 * 7 + j];
        for (int j = 0; j < 9; j++)
            link6_rot_mat[j] = d->geom_xmat[9 * 7 + j];
        mju_mat2Quat(link6_rot_quat, link6_rot_mat);

        //e = link6_pos - ref_link6_pos
        for (int j = 0; j < 3; j++)
            e[j] = link6_pos[j] - ref_link6_pos[j];
        for (int j = 0; j < 3; j++)
            e[j + 3] = (link6_rot_quat[j + 1] - ref_link6_rot[j + 1]);
        //calculate jacobian
        /*mj_jacGeom(m, d, jacp, jacr, int(7));
        //mj_jac(m, d, jacp, jacr, link6_pos,int(7));
        for (int j = 0; j < 24; j++)
            Full_jacobian[j] = jacp[j];
        for (int j = 0; j < 24; j++)
            Full_jacobian[j + 24] = jacr[j];*/
        eef_full_jacobian(Full_jacobian);
//        eef_full_jacobian(Full_jacobian_global);
//        cout << "jacBody" << endl;
//        mju_printMat(Full_jacobian_global, 6, 8);

        //cartesian link 6 vel = Full_jacobian * qvel
        mju_mulMatVec(ref_link6_vel, Full_jacobian, d->qvel, 6, m->nv);
        //naive PD control law
        for (int j = 0; j < 6; j++)
            force[j] = -kp[j] * e[j] - kd[j] * ref_link6_vel[j];
        //transform cartesian force to joint space torque
        mju_mulMatTVec(force_torque, Full_jacobian, force, 6, m->nv);

        if (true)
        {
            cout << "################ Control variables ###################" << endl;
            cout << "link6_pos:" << endl;
            mju_printMat(link6_pos, 1, 3);
            cout << "link6_rot_quat:" << endl;
            mju_printMat(link6_rot_quat, 1, 4);
            cout << "error:" << endl;
            mju_printMat(e, 1, 6);
            cout << "force:" << endl;
            mju_printMat(force, 1, 6);
            cout << "force_torque:" << endl;
            mju_printMat(force_torque, 1, 8);
            cout << "jacBody" << endl;
            mju_printMat(jacp, 3, 8);
            mju_printMat(jacr, 3, 8);
            mju_printMat(Full_jacobian, 6, 8);
        }

//        mj_mulM(m, d, inertial_torque, ref_acc);
//        mju_add(d->ctrl, inertial_torque, d->qfrc_inverse, 8);
//        mju_addTo(d->ctrl, force_torque, 8);
        mjtNum M[m->nv*m->nv];
	    mjtNum M_robot[m->nu*m->nu];
	    mj_fullM(m, M, d->qM);
	    extract_submatrix(M, M_robot, 0, m->nu, 0, m->nu);
	    mju_mulMatVec(inertial_torque, M_robot, ref_acc, 8, 8);

        mju_add(d->ctrl, inertial_torque, d->qfrc_inverse, 8);
        mju_addTo(d->ctrl, force_torque, 8);

        // set gripper ctrl
        d->ctrl[6] = gripper_pose;
        d->ctrl[7] = gripper_pose;
    }
}

void apply_FT_on_eef(const mjModel *m, mjData *d, mjtNum *force)
{
    //apply a six dimensional force & torque on the end effector
    if (m->nu == m->nv)
    {
        //calculate qfrc_inverse
        mj_rne(m, d, 0, d->qfrc_inverse);
        for (int i = 0; i < m->nv; i++)
            d->qfrc_inverse[i] += m->dof_armature[i] * d->qacc[i] - d->qfrc_passive[i] - d->qfrc_constraint[i];

        mjtNum inertial_torque[8];
        mjtNum force[6];
        mjtNum force_torque[8]; //torque transformed from force
        mjtNum jacp[24];
        mjtNum jacr[24];
        mjtNum Full_jacobian[48];

        //calculate jacobian
        mj_jacGeom(m, d, jacp, jacr, int(7));
        for (int j = 0; j < 24; j++)
            Full_jacobian[j] = jacp[j];
        for (int j = 0; j < 24; j++)
            Full_jacobian[j + 24] = jacr[j];

        //transform cartesian space force to joint space torque
        mju_mulMatTVec(force_torque, Full_jacobian, force, 6, 8);
        mj_mulM(m, d, inertial_torque, ref_acc);
        mju_add(d->ctrl, inertial_torque, d->qfrc_inverse, 8);
        mju_addTo(d->ctrl, force_torque, 8);
        if (Verbose)
        {
            cout << "################ Force applied ###################" << endl;
            cout << "force:" << endl;
            mju_printMat(force, 1, 6);
            cout << "force_torque:" << endl;
            mju_printMat(force_torque, 1, 8);
            //cout << "jacBody" << endl;
            //mju_printMat(jacp, 3, 8);
            //mju_printMat(jacr, 3, 8);
            //mju_printMat(Full_jacobian, 6, 8);
        }
        // set gripper ctrl
        d->ctrl[6] = gripper_pose;
        d->ctrl[7] = gripper_pose;
    }
}

void close(void)
{
    glfwDestroyWindow(window);
    //free visualization storage
    mjv_freeScene(&scn);
    mjr_freeContext(&con);

    // free MuJoCo model and data, deactivate
    mj_deleteData(d);
    mj_deleteModel(m);
    mj_deactivate();

// terminate GLFW (crashes with Linux NVidia drivers)
#if defined(__APPLE__) || defined(_WIN32)
    glfwTerminate();
#endif
    cout << "Have a nice day :)" << endl;
}

void update_reference_traj(mjtNum *tar_joint, mjtNum *tar_vel, mjtNum *tar_acc)
{
    mju_copy(ref_joint, tar_joint, 8);
    mju_copy(ref_vel, tar_vel, 8);
    mju_copy(ref_acc, tar_acc, 8);
}

void update_pd_gain(mjtNum *new_kp, mjtNum *new_kv)
{
    for (int i = 0; i < 6; i++)
    {
        kp[i + i * 8] = new_kp[i];
        kv[i + i * 8] = new_kv[i];
    }

}

void update_pi_gain(mjtNum *new_kp, mjtNum *new_ki)
{
    for (int i = 0; i < 6; i++)
    {
        kp[i + i * 8] = new_kp[i];
        ki[i + i * 8] = new_ki[i];
    }

}

void update_gripper_state(double gripper_p)
{
    gripper_pose = gripper_p;
}

void step(void)
{
    // advance interactive simulation for 1/60 sec
    //  Assuming MuJoCo can simulate faster than real-time, which it usually can,
    //  this loop will finish on time for the next frame to be rendered at 60 fps.
    //  Otherwise add a cpu timer and exit this loop when it is time to render.
    if (d->time - timer_render > 1.0 / 60.0 && Render)
        update_ui(window);

    // step frequency about 125
    mj_step1(m, d);
    switch (controller)
    {
    case 0:
        computed_torque_control(m, d);
        break;
    case 1:
        cartesian_impedance_control(m, d);
        break;
    case 2:
        mjtNum force[6];
        apply_FT_on_eef(m, d, force);
        break;
    case 3:
        joint_vel_control(m, d);
        break;
    }
    mj_step2(m, d);
    if (Render){
        if (glfwWindowShouldClose(window))
            close();
    }
}


// initalize the simulation and the ui
void init(void)
{
    // print version, check compatibility
    cout << "MuJoCo Pro library version " << 0.01 * mj_version() << endl;
    if (mjVERSION_HEADER != mj_version())
        mju_error("Headers and library have different versions");
    char *license_path = getenv("MUJOCO_LICENSE_PATH");
    mj_activate(license_path); // Check MuJoCo activation license

    char error[1000];
    m = mj_loadXML("./impedance_envs/impedance_envs/envs/source/LRMate_200iD.xml", 0, error, 1000); // MuJoCo model
    d = mj_makeData(m);                                        // MuJoCo data

    if (Render){
        if (!glfwInit())
            mju_error("Could not initialize GLFW"); // init GLFW

        // create window, make OpenGL context current, request v-sync
        window = glfwCreateWindow(1800, 1200, "Fanuc Robot Simulation", NULL, NULL);
        glfwMakeContextCurrent(window);
        glfwSwapInterval(1);
        // initialize visualization data structures
        mjv_defaultCamera(&cam);
        mjv_defaultOption(&opt);
        mjv_defaultScene(&scn);
        mjr_defaultContext(&con);

        // create scene and context
        mjv_makeScene(m, &scn, 2000);
        mjr_makeContext(m, &con, mjFONTSCALE_150);

        // install GLFW mouse and keyboard callbacks
        glfwSetKeyCallback(window, keyboard);
        glfwSetCursorPosCallback(window, mouse_move);
        glfwSetMouseButtonCallback(window, mouse_button);
        glfwSetScrollCallback(window, scroll);
    }
    timer_render = d->time;

    // set up a controller, we'll not use this as we want to have more freedoms of the control loop
    // mjcb_control = computed_torque_control;
}

int main(void)
{
    init();
    /*mjtNum np[6] = {17, 17, 17, 17, 17, 17};
    mjtNum nv[6] = {40, 40, 40, 40, 40, 40};
    update_pd_gain(np, nv);*/

    mjtNum np[6] = {17, 17, 17, 17, 17, 17};
    mjtNum ni[6] = {1, 1, 1, 1, 1, 1};
    update_pi_gain(np, ni);

    mjtNum j[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    mjtNum v[8] = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0};
    mjtNum a[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    update_reference_traj(j, v, a);
    
    // run main loop, target real-time simulation and 60 fps rendering
    while (!glfwWindowShouldClose(window))
    {
        step();
    }
    close();
    return 1;
}

extern "C"
{
    void wrapper_init(void)
    {
        init();
        if(Render){
            update_ui(window);
        }
    }

    void wrapper_set_verbose(bool v)
    {
        Verbose = v;
    }

    void wrapper_set_render(bool r)
    {
        Render = r;
    }

    void wrapper_update_reference_traj(mjtNum *tar_joint, mjtNum *tar_vel, mjtNum *tar_acc)
    {
        update_reference_traj(tar_joint, tar_vel, tar_acc);
    }

    void wrapper_update_pd_gain(mjtNum *new_kp, mjtNum *new_kv)
    {
        update_pd_gain(new_kp, new_kv);
    }

    void wrapper_update_gripper_state(double gripper_p)
    {
        update_gripper_state(gripper_p);
    }

    void wrapper_Euler2quat(mjtNum *Euler, mjtNum *quat)
    {
        Euler2quat(Euler, quat);
    }

    void wrapper_step(void)
    {
        step();
    }

    void wrapper_get_sim_time(mjtNum *time)
    {
        time[0] = d->time;
    }

    void wrapper_update_ui(void)
    {
        update_ui(window);
    }

    void wrapper_close(void)
    {
        close();
    }

    void wrapper_reset(void)
    {
        mj_resetData(m, d);
        mj_forward(m, d);
    }

    void wrapper_get_joint_states(mjtNum *joint_pose, mjtNum *joint_vel, mjtNum *joint_acc)
    {
        mju_copy(joint_pose, d->qpos, 8);
        mju_copy(joint_vel, d->qvel, 8);
        mju_copy(joint_acc, d->qacc, 8);
    }

    void wrapper_set_joint_states(mjtNum *joint_pose, mjtNum *joint_vel, mjtNum *joint_acc)
    {
        mju_copy(d->qpos, joint_pose, 8);
        mju_copy(d->qvel, joint_vel,  8);
        mju_copy(d->qacc, joint_acc,  8);
    }

    void wrapper_get_sensor_reading(mjtNum *Force)
    {
        int sensorId = mj_name2id(m, mjOBJ_SENSOR, "force_ee");
        int adr = m->sensor_adr[sensorId];
        int dim = m->sensor_dim[sensorId];
//        mjtNum sensor_data[dim];
        mju_copy(Force, &d->sensordata[adr], dim);
    }

    void wrapper_nv(mjtNum *nv)
    {
//        mju_copy(nv, mjtNum(m->nv), 1);
        nv[0] = mjtNum(m->nv);
    }

    void wrapper_eef_full_jacobian(mjtNum *Full_jacobian)
    {
        mjtNum jacp[3*m->nv];
        mjtNum jacr[3*m->nv];
        mj_jacGeom(m, d, jacp, jacr, int(7));
        for (int j = 0; j < 3*m->nv; j++)
            Full_jacobian[j] = jacp[j];
        for (int j = 0; j < 3*m->nv; j++)
            Full_jacobian[j + 3*m->nv] = jacr[j];
    }

    void wrapper_update_controller_type(int controller_idx)
    {
        update_controller_type(controller_idx);
    }

    void get_eef_pose_vel(mjtNum *pose_vel)
    {
        mjtNum link6_pos[3];
        mjtNum link6_rot_mat[9];
        mjtNum link6_rot_quat[4];
        mjtNum jacp[24];
        mjtNum jacr[24];
        mjtNum Full_jacobian[48];
        mjtNum link6_vel[6];
        int geom_id = 7;
        for (int j = 0; j < 3; j++)
            link6_pos[j] = d->geom_xpos[3 * geom_id + j];
        for (int j = 0; j < 9; j++)
            link6_rot_mat[j] = d->geom_xmat[9 * geom_id + j];
        mju_mat2Quat(link6_rot_quat, link6_rot_mat);

        //calculate jacobian
        mj_jacGeom(m, d, jacp, jacr, int(geom_id));
        //mj_jac(m, d, jacp, jacr, link6_pos,int(7));
        for (int j = 0; j < 24; j++)
            Full_jacobian[j] = jacp[j];
        for (int j = 0; j < 24; j++)
            Full_jacobian[j + 24] = jacr[j];

        //cartesian link 6 vel = Full_jacobian * qvel
        mju_mulMatVec(link6_vel, Full_jacobian, d->qvel, 6, 8);
        for (int j = 0; j < 3; j++)
            pose_vel[j] = link6_pos[j];
        for (int j = 0; j < 3; j++)
            pose_vel[j + 3] = link6_rot_quat[j];
        for (int j = 0; j < 6; j++)
            pose_vel[j + 7] = link6_vel[j];
    }
}