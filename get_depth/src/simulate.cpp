//-----------------------------------//
//  This file is part of MuJoCo.     //
//  Copyright 2009-2015 Roboti LLC.  //
//-----------------------------------//

#include "mujoco.h"
#include "glfw3.h"
 
#include <GL/glut.h> 

#include "stdlib.h"
#include "string.h"
#include <mutex>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//#include <opencv2/contrib/contrib.hpp> // opencv2, not 3.0


#include <stdio.h>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <unistd.h>

//-------------------------------- global variables -------------------------------------

// syncrhonization
std::mutex gui_mutex;

// model
mjModel* m = 0;
mjData* data = 0;
char lastfile[100] = "";

// user state
bool paused = true;
bool showoption = false;
bool showinfo = true;
bool showdepth = false;
int showhelp = 1;                   // 0: none; 1: brief; 2: full
int speedtype = 1;                  // 0: slow; 1: normal; 2: max

bool start = false;

// abstract visualization
mjvObjects objects;
mjvCamera cam;
mjvOption vopt;
char status[1000] = "";

// OpenGL rendering
mjrContext con;
mjrOption ropt;
double scale = 1;
bool stereoavailable = false;
int Wdepth = 640;
int Hdepth = 480;
float depth_buffer[640*480];       // big enough for 4K screen
unsigned char color_buffer[640*480*3];       // big enough for 4K screen

unsigned char depth_rgb[960*540*3];  // 1/4th of screen
double zNear = 0.01;
double zFar = 50.0;
        
// selection and perturbation
bool button_left = false;
bool button_middle = false;
bool button_right =  false;
int lastx = 0;
int lasty = 0;
int selbody = 0;
int perturb = 0;
mjtNum selpos[3] = {0, 0, 0};
mjtNum refpos[3] = {0, 0, 0};
mjtNum refquat[4] = {1, 0, 0, 0};
int needselect = 0;                 // 0: none, 1: select, 2: center 

int contdepth = 0;

///projection matrix

double projection[16];
// help strings
const char help_title[] = 
"Help\n"
"Option\n"
"Info\n"
"Depth map\n"
"Stereo\n"
"Speed\n"
"Pause\n"
"Reset\n"
"Forward\n"
"Back\n"
"Forward 100\n"
"Back 100\n"
"Autoscale\n"
"Reload\n"
"Geoms\n"
"Sites\n"
"Select\n"
"Center\n"
"Zoom\n"
"Camera\n"
"Perturb\n"
"Switch Cam";

const char help_content[] = 
"F1\n"
"F2\n"
"F3\n"
"F4\n"
"F5\n"
"Enter\n"
"Space\n"
"BackSpace\n"
"Right arrow\n"
"Left arrow\n"
"Page Down\n"
"Page Up\n"
"Ctrl A\n"
"Ctrl L\n"
"0 - 4\n"
"Shift 0 - 4\n"
"L double-click\n"
"R double-click\n"
"Scroll or M drag\n"
"[Shift] L/R drag\n"
"Ctrl [Shift] drag\n"
"[ ]";

char opt_title[1000] = "";
char opt_content[1000];

std::string image_dir;

//-------------------------------- utility functions ------------------------------------
void projectionToOpenGLTopLeft( double p[16], const cv::Mat &K, int w, int h, float zNear, float zFar)
{
    
    float fu = K.at<double>(0,0);
    float fv = (float)K.at<double>(1, 1);
    float u0 = (float)K.at<double>(0, 2);
    float v0 = (float)K.at<double>(1, 2);    

    float L = -(u0) * zNear / fu;
    float R = +(w-u0) * zNear / fu;
    float T = -(v0) * zNear / fv;
    float B = +(h-v0) * zNear / fv;   
        
    std::fill_n(p,4*4,0);
    
    p[0*4+0] = 2 * zNear / (R-L);
    p[1*4+1] = 2 * zNear / (T-B);    
    p[2*4+0] = (R+L)/(L-R);
    p[2*4+1] = (T+B)/(B-T);
    p[2*4+2] = (zFar +zNear) / (zFar - zNear);
    p[2*4+3] = 1.0;    
    p[3*4+2] =  (2*zFar*zNear)/(zNear - zFar);

}

void
colorImage(cv::Mat img_depth, cv::Mat *falseColorsMap){
    
        double min;
        double max;
        cv::minMaxIdx(img_depth, &min, &max);
        printf("max depth = %lf\n", max);
        printf("min depth = %lf\n", min);
       
        
        cv::Mat adjMap;
        // expand your range to 0..255. Similar to histEq();
        img_depth.convertTo(adjMap,CV_8UC1, 255 / (max-min), -min); 

        // this is great. It converts your grayscale image into a tone-mapped one, 
        // much more pleasing for the eye
        // function is found in contrib module, so include contrib.hpp 
        // and link accordingly        
        applyColorMap(adjMap, *falseColorsMap, cv::COLORMAP_JET);

        
}

// center and scale view
void autoscale(GLFWwindow* window)
{
    // autoscale
    cam.lookat[0] = m->stat.center[0];
    cam.lookat[1] = m->stat.center[1];
    cam.lookat[2] = m->stat.center[2];
    cam.distance = 1.5 * m->stat.extent;
    cam.camid = -1;
    cam.trackbodyid = -1;
    if( window )
    {
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);        
        mjv_updateCameraPose(&cam, (mjtNum)width/(mjtNum)height);
    }
}



// load mjb or xml model
void loadmodel(GLFWwindow* window, const char* filename)
{
    // load and compile
    char error[500] = "could not load binary model";
    mjModel* mnew = 0;
    if( strlen(filename)>4 && !strcmp(filename+strlen(filename)-4, ".mjb") )
        mnew = mj_loadModel(filename, 0, 0);
    else
        mnew = mj_loadXML(filename, error);
    if( !mnew )
    {
        printf("%s\n", error);
        return;
    }

    // delete old model, assign new
    mj_deleteData(data);
    mj_deleteModel(m);
    m = mnew;
    data = mj_makeData(m);
    mj_forward(m, data);

    // save filename for reload
    strcpy(lastfile, filename);

    // re-create custom context
    mjr_makeContext(m, &con, 150);
    
   
    // clear perturbation state
    perturb = 0;
    selbody = 0;
    needselect = 0;

    // set title
    if( window && m->names )
        glfwSetWindowTitle(window, m->names);

    // center and scale view
    autoscale(window);
}


//--------------------------------- callbacks -------------------------------------------

// keyboard
void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods)
{
    int n;

    // require model
    if( !m )
        return;

    // do not act on release
    if( act==GLFW_RELEASE )
        return;

    gui_mutex.lock();

    switch( key )
    {
    case GLFW_KEY_F1:                   // help
        showhelp++;
        if( showhelp>2 )
            showhelp = 0;
        break;

    case GLFW_KEY_F2:                   // option
        showoption = !showoption;
        break;

    case GLFW_KEY_F3:                   // info
        showinfo = !showinfo;
        break;

    case GLFW_KEY_F4:                   // depthmap
        showdepth = !showdepth;
        break;

    case GLFW_KEY_F5:                   // stereo
        if( stereoavailable )
            ropt.stereo = !ropt.stereo;
        break;

    case GLFW_KEY_ENTER:                // speed
        speedtype += 1;
        if( speedtype>2 )
            speedtype = 0;
        break;

    case GLFW_KEY_SPACE:                // pause
        paused = !paused;
        break;

    case GLFW_KEY_BACKSPACE:            // reset
        mj_resetData(m, data);
        mj_forward(m, data);
        break;

    case GLFW_KEY_RIGHT:                // step forward
        if( paused )
            mj_step(m, data);
        break;

    case GLFW_KEY_LEFT:                 // step back
        if( paused )
        {
            m->opt.timestep = -m->opt.timestep;
            mj_step(m, data);
            m->opt.timestep = -m->opt.timestep;
        }
        break;

    case GLFW_KEY_PAGE_DOWN:            // step forward 100
        if( paused )
            for( n=0; n<100; n++ )
                mj_step(m,data);
        break;

    case GLFW_KEY_PAGE_UP:              // step back 100
        if( paused )
        {
            m->opt.timestep = -m->opt.timestep;
            for( n=0; n<100; n++ )
                mj_step(m,data);
            m->opt.timestep = -m->opt.timestep;
        }
        break;

    case GLFW_KEY_LEFT_BRACKET:         // previous camera
        if( cam.camid>-1 )
            cam.camid--;
        break;

    case GLFW_KEY_RIGHT_BRACKET:        // next camera
        if( cam.camid<m->ncam-1 )
            cam.camid++;
        break;
    
    case GLFW_KEY_Y:
        //****** start recording *******/
        start = true;
        showdepth = true;
        break;

    default:
        // control keys
        if( mods & GLFW_MOD_CONTROL )
        {
            if( key==GLFW_KEY_A )
                autoscale(window);
            else if( key==GLFW_KEY_L && lastfile[0] )
                loadmodel(window, lastfile);
			else if ( key==GLFW_KEY_W )
				glfwSetWindowShouldClose(window, GL_TRUE);
            else if ( key==GLFW_KEY_P ) {
                std::cout << "Qpos: ";
                for (int i=0; i<m->nq; i++) { // joint positions
                    std::cout<<data->qpos[i]<<" ";
                }
                std::cout <<"\nQvel: ";
                for (int i=0; i<m->nv; i++) { // velocities
                    std::cout<<data->qvel[i]<<" ";
                }
                std::cout <<"\nQacc: ";
                for (int i=0; i<m->nv; i++) { // accelerations
                    std::cout<<data->qacc[i]<<" ";
                }
                std::cout <<"\nCtrl: ";
                for (int i=0; i<m->nu; i++) { // controls
                    std::cout<<data->ctrl[i]<<" ";
                }
                std::cout<<"\n";
            }

            break;
        }

        // toggle visualization flag
        for( int i=0; i<mjNVISFLAG; i++ )
            if( key==mjVISSTRING[i][2][0] )
                vopt.flags[i] = !vopt.flags[i];

        // toggle rendering flag
        for( int i=0; i<mjNRNDFLAG; i++ )
            if( key==mjRNDSTRING[i][2][0] )
                ropt.flags[i] = !ropt.flags[i];

        // toggle geom/site group
        for( int i=0; i<mjNGROUP; i++ )
            if( key==i+'0')
            {
                if( mods & GLFW_MOD_SHIFT )
                    vopt.sitegroup[i] = !vopt.sitegroup[i];
                else
                    vopt.geomgroup[i] = !vopt.geomgroup[i];
            }
    }
    
    gui_mutex.unlock();
}


// mouse button
void mouse_button(GLFWwindow* window, int button, int act, int mods)
{
    // past data for double-click detection
    static int lastbutton = 0;
    static double lastclicktm = 0;

    // update button state
    button_left =   (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS);
    button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE)==GLFW_PRESS);
    button_right =  (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT)==GLFW_PRESS);

    // update mouse position
    double x, y;
    glfwGetCursorPos(window, &x, &y);
    lastx = (int)(scale*x);
    lasty = (int)(scale*y);

    // require model
    if( !m )
        return;

    gui_mutex.lock();

    // set perturbation
    int newperturb = 0;
    if( (mods & GLFW_MOD_CONTROL) && selbody>0 )
    {
        // right: translate;  left: rotate
        if( button_right )
            newperturb = mjPERT_TRANSLATE;
        else if( button_left )
            newperturb = mjPERT_ROTATE;

        // perturbation onset: reset reference
        if( newperturb && !perturb )
        {
            int id = paused ? m->body_rootid[selbody] : selbody;
            mju_copy3(refpos, data->xpos+3*id);
            mju_copy(refquat, data->xquat+4*id, 4);
        }
    }
    perturb = newperturb;

    // detect double-click (250 msec)
    if( act==GLFW_PRESS && glfwGetTime()-lastclicktm<0.25 && button==lastbutton )
    {
        if( button==GLFW_MOUSE_BUTTON_LEFT )
            needselect = 1;
        else
            needselect = 2;

        // stop perturbation on select
        perturb = 0;
    }

    // save info
    if( act==GLFW_PRESS )
    {
        lastbutton = button;
        lastclicktm = glfwGetTime();
    }

    gui_mutex.unlock();
}


// mouse move
void mouse_move(GLFWwindow* window, double xpos, double ypos)
{
    // no buttons down: nothing to do
    if( !button_left && !button_middle && !button_right )
        return;

    // compute mouse displacement, save
    float dx = (int)(scale*xpos) - (float)lastx;
    float dy = (int)(scale*ypos) - (float)lasty;
    lastx = (int)(scale*xpos);
    lasty = (int)(scale*ypos);

    // require model
    if( !m )
        return;

    // get current window size
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    // get shift key state
    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)==GLFW_PRESS ||
                      glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT)==GLFW_PRESS);

    // determine action based on mouse button
    mjtMouse action;
    if( button_right )
        action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    else if( button_left )
        action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    else
        action = mjMOUSE_ZOOM;

    gui_mutex.lock();

    // perturbation
    if( perturb )
    {
        if( selbody>0 )
            mjv_moveObject(action, dx, dy, &cam.pose, 
                           (float)width, (float)height, refpos, refquat);
    }

    // camera control
    else
        mjv_moveCamera(action, dx, dy, &cam, (float)width, (float)height);


    
    gui_mutex.unlock();
}


// scroll
void scroll(GLFWwindow* window, double xoffset, double yoffset)
{
    // require model
    if( !m )
        return;

    // get current window size
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    // scroll
    gui_mutex.lock();
    mjv_moveCamera(mjMOUSE_ZOOM, 0, (float)(-20*yoffset), &cam, (float)width, (float)height);
    gui_mutex.unlock();
}


// drop
void drop(GLFWwindow* window, int count, const char** paths)
{
    // make sure list is non-empty
    if( count>0 )
    {
        gui_mutex.lock();
        loadmodel(window, paths[0]);
        gui_mutex.unlock();
    }
}


//-------------------------------- simulation and rendering -----------------------------

// make option string
void makeoptionstring(const char* name, char key, char* buf)
{
    int i=0, cnt=0;

    // copy non-& characters
    while( name[i] && i<50 )
    {
        if( name[i]!='&' )
            buf[cnt++] = name[i];

        i++;
    }

    // finish
    buf[cnt] = ' ';
    buf[cnt+1] = '(';
    buf[cnt+2] = key;
    buf[cnt+3] = ')';
    buf[cnt+4] = 0;
}


// advance simulation
void advance(void)
{
    // perturbations
    if( selbody>0 )
    {
        // fixed object: edit
        if( m->body_jntnum[selbody]==0 && m->body_parentid[selbody]==0 )
            mjv_mouseEdit(m, data, selbody, perturb, refpos, refquat);
    
        // movable object: set mouse perturbation
        else
            mjv_mousePerturb(m, data, selbody, perturb, refpos, refquat, 
                             data->xfrc_applied+6*selbody);
    }

    // advance simulation
    mj_step(m, data);

    // clear perturbation
    if( selbody>0 )
        mju_zero(data->xfrc_applied+6*selbody, 6);
}


// render
void render(GLFWwindow* window)
{
    // past data for FPS calculation
    static double lastrendertm = 0;
    float symcycle = 0.015;
    // get current window rectangle
    mjrRect rect = {0, 0, 0, 0};
    glfwGetFramebufferSize(window, &rect.width, &rect.height);

    double duration = 0;
    gui_mutex.lock();

    // no model: empty screen
    if( !m )
    {
		mjr_rectangle(1, rect, 0, 0, rect.width, rect.height, 
			0.2, 0.3, 0.4, 1);
        mjr_overlay(rect, mjGRID_TOPLEFT, 0, "Drag-and-drop model file here", 0, &con);
        gui_mutex.unlock();
        return;
    }

    // start timers
    double starttm = glfwGetTime();
    mjtNum startsimtm = data->time;

    // paused
    if( paused )
    {
        // edit
        mjv_mouseEdit(m, data, selbody, perturb, refpos, refquat);

        // recompute to refresh rendering
        mj_forward(m, data);

        // 15 msec delay
        while( glfwGetTime()-starttm<symcycle );
    }

    // running
    else
    {
        // simulate for 15 msec of CPU time
        int n = 0;
        while( glfwGetTime()-starttm<symcycle )
        {
            // step at specified speed
            if( (speedtype==0 && n==0) || (speedtype==1 && data->time-startsimtm<0.016) || speedtype==2 )
            {
                advance();
                n++;
            }

            // simulation already done: compute duration
            else if( duration==0 && n )
                duration = 1000*(glfwGetTime() - starttm)/n;

        }

        // compute duration if not already computed
        if( duration==0 && n )
            duration = 1000*(glfwGetTime() - starttm)/n;
    }

    // update simulation statistics
    if( !paused )
        sprintf(status, "%.1f\n%d (%d)\n%.2f\n%.0f          \n%.2f\n%.2f\n%d",
                data->time, data->nefc, data->ncon, 
                duration, 1.0/(glfwGetTime()-lastrendertm),
                data->energy[0]+data->energy[1],
                mju_log10(mju_max(mjMINVAL,
                                  mju_abs(data->solverstat[0]-data->solverstat[1]) /
                                  mju_max(mjMINVAL,mju_abs(data->solverstat[0])+mju_abs(data->solverstat[1])))),
                cam.camid );
    lastrendertm = glfwGetTime();

    // create geoms and lights
    mjv_makeGeoms(m, data, &objects, &vopt, mjCAT_ALL, selbody, 
                  (perturb & mjPERT_TRANSLATE) ? refpos : 0, 
                  (perturb & mjPERT_ROTATE) ? refquat : 0, selpos); 
    mjv_makeLights(m, data, &objects);

    // update camera
   mjv_setCamera(m, data, &cam);
   mjv_updateCameraPose(&cam, (mjtNum)rect.width/(mjtNum)rect.height);

    
    // selection
    if( needselect )
    {
        // find selected geom
        mjtNum pos[3];
        int selgeom = mjr_select(rect, &objects, lastx, rect.height - lasty, 
                                 pos, 0, &ropt, &cam.pose, &con);

        // set lookat point
        if( needselect==2 )
        {
            if( selgeom >= 0 )
                mju_copy3(cam.lookat, pos);
        }

        // set body selection
        else
        {
            if( selgeom>=0 && objects.geoms[selgeom].objtype==mjOBJ_GEOM )
            {
                // record selection
                selbody = m->geom_bodyid[objects.geoms[selgeom].objid];

                // clear if invalid
                if( selbody<0 || selbody>=m->nbody )
                    selbody = 0;

                // otherwise compute selpos
                else
                {
                    mjtNum tmp[3];
                    mju_sub3(tmp, pos, data->xpos+3*selbody);
                    mju_mulMatTVec(selpos, data->xmat+9*selbody, tmp, 3, 3);
                }
            }
            else
                selbody = 0;
        }

        needselect = 0;
    }

     //printf("camera focal lenght y:  [%lf ]\n", cam.fovy  );
     //printf("z to the object:  [%lf ]\n", cam.distance);
     //printf("cam azimuth:  [%lf ]\n", cam.azimuth);
     //printf("cam elevation:  [%lf ]\n", cam.elevation);
     //printf("cam lookat:  [%lf %lf %lf  ]\n", cam.lookat[0],cam.lookat[1],cam.lookat[2]);
     //
     //printf("\ncontext zfar = %lf \n",con.zfar); 
     //printf("context znear = %lf \n",con.znear); 
     
    
    // render rgb
    mjr_render(0, rect, &objects, &ropt, &cam.pose, &con);

    
    //printf("Camera pose scale: %lf\n", cam.pose.scale);
    // show depth map
    if( showdepth )
    {
        // get the depth buffer
        mjr_getBackbuffer(color_buffer, depth_buffer, rect, &con);
        
        /********* save the images ***********/
        cv::Mat depth_im(Hdepth,Wdepth,CV_32F,(float *)depth_buffer);
        cv::Mat color_im(Hdepth,Wdepth,CV_8UC3,(uchar *)color_buffer);
        
        cv::Mat depth_mm(Hdepth,Wdepth,CV_16U,cv::Scalar(0));
        cv::Mat depth_flip;
        cv::flip(depth_im,depth_flip,0);
        cv::flip(color_im,color_im,0);
        cv::cvtColor(color_im,color_im,CV_RGB2BGR);
        for(int i=0; i <Hdepth;i++) {
            for(int j=0; j <Wdepth;j++)
            {
                float z_b = depth_flip.at<float>(i,j);
                float z_n = 2.0 * z_b - 1.0;
                float z_e = 2.0 * zNear * zFar / (zFar + zNear - z_n * (zFar - zNear));
                float depth = z_e  +  zNear;
                if (depth < zFar)
                    depth_mm.at<short>(i,j) = (short) (depth * 1000.0); 
                else
                    depth_mm.at<short>(i,j) = (short) (0.0); 
            }
        }
      //  cv::flip(depth_mm,depth_mm,0);
        
        std::stringstream ss;
        ss << image_dir+"/depth/" << std::setfill('0') << std::setw(4) << contdepth << ".png";
        cv::imwrite(ss.str(), depth_mm);
        cv::Mat image = cv::imread(ss.str(),CV_LOAD_IMAGE_ANYDEPTH);
        image.convertTo(image,CV_32F);
        cv::Mat depth_color;
//         colorImage(image,&depth_color);
//         cv::imshow("imag", depth_color);
//         cvWaitKey(33);
        ss.str(""); ss.clear();
        ss << image_dir+"/color/" << std::setfill('0') << std::setw(4) << contdepth << ".png";
        cv::imwrite(ss.str(), color_im);
        
       
        contdepth++;
        // convert to RGB, subsample by 4
        for( int r=0; r<rect.height; r+=4 )
            for( int c=0; c<rect.width; c+=4 )
            {
                // get subsampled address
                int adr = (r/4)*(rect.width/4) + c/4;

                // assign rgb
                depth_rgb[3*adr] = depth_rgb[3*adr+1] = depth_rgb[3*adr+2] = 
                    (unsigned char)((1.0f-depth_buffer[r*rect.width+c])*255.0f);
            }

        // show in bottom-right corner
        mjr_showBuffer(depth_rgb, rect.width/4, rect.height/4, (3*rect.width)/4, 0, &con);
    }

    // show overlays
    if( showhelp==1 )
        mjr_overlay(rect, mjGRID_TOPLEFT, 0, "Help  ", "F1  ", &con);
    else if( showhelp==2 )
        mjr_overlay(rect, mjGRID_TOPLEFT, 0, help_title, help_content, &con);

    if( showinfo )
    {
        if( paused )
            mjr_overlay(rect, mjGRID_BOTTOMLEFT, 0, "PAUSED", 0, &con);
        else
            mjr_overlay(rect, mjGRID_BOTTOMLEFT, 0, 
                "Time\nSize\nCPU\nFPS\nEngy\nStat\nCam", status, &con);
    }

    if( showoption )
    {
        int i;
        char buf[100];

        // fill titles on first pass
        if( !opt_title[0] )
        {
            for( i=0; i<mjNRNDFLAG; i++)
            {
                makeoptionstring(mjRNDSTRING[i][0], mjRNDSTRING[i][2][0], buf);
                strcat(opt_title, buf);
                strcat(opt_title, "\n");
            }
            for( i=0; i<mjNVISFLAG; i++)
            {
                makeoptionstring(mjVISSTRING[i][0], mjVISSTRING[i][2][0], buf);
                strcat(opt_title, buf);
                if( i<mjNVISFLAG-1 )
                    strcat(opt_title, "\n");
            }
        }

        // fill content
        opt_content[0] = 0;
        for( i=0; i<mjNRNDFLAG; i++)
        {
            strcat(opt_content, ropt.flags[i] ? " + " : "   ");
            strcat(opt_content, "\n");
        }
        for( i=0; i<mjNVISFLAG; i++)
        {
            strcat(opt_content, vopt.flags[i] ? " + " : "   ");
            if( i<mjNVISFLAG-1 )
                strcat(opt_content, "\n");
        }

        // show
        mjr_overlay(rect, mjGRID_TOPRIGHT, 0, opt_title, opt_content, &con);
    }

    gui_mutex.unlock();
}


//-------------------------------- main function ----------------------------------------

int main(int argc, const char** argv)
{
    // activate MuJoCo license
    mj_activate("mjkey.txt");

    // init GLFW, set multisampling
    if (!glfwInit())
        return 1;
    glfwWindowHint(GLFW_SAMPLES, 4);

    // try stereo if refresh rate is at least 100Hz
    GLFWwindow* window = 0;
    if( glfwGetVideoMode(glfwGetPrimaryMonitor())->refreshRate>=100 )
    {
        glfwWindowHint(GLFW_STEREO, 1);
        window = glfwCreateWindow(1200, 900, "Simulate", NULL, NULL);
        if( window )
            stereoavailable = true;
    }

    // no stereo: try mono
    if( !window )
    {
        glfwWindowHint(GLFW_STEREO, 0);
        window = glfwCreateWindow(640, 480, "Simulate", NULL, NULL);
    }
    if( !window )
    {
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);

    // determine retina scaling
    int width, width1, height;
    glfwGetFramebufferSize(window, &width, &height);
    glfwGetWindowSize(window, &width1, &height);
    scale = (double)width/(double)width1;
   
    
    // init MuJoCo rendering
    mjv_makeObjects(&objects, 1000);
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&vopt);
    mjr_defaultOption(&ropt);
    mjr_defaultContext(&con);    
  
    mjr_makeContext(m, &con, 150);

    
    
    // load model if filename given as argument
    // context is defined here
    if( argc==2 )
        loadmodel(window, argv[1]);
    else
        printf("Specify which model to load\n");
    
    if (!m) {
        printf("Did not load model\n");
        return 1;
    }
    con.zfar = zFar;
    con.znear = zNear;
    cam.fovy = 45.0;

    printf("context zfar = %lf \n",con.zfar); 
    printf("context znear = %lf \n",con.znear); 
    
//     con.zfar = 24.275 ;
//     con.znear = 0.0971;

    // set GLFW callbacks
    glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);
    glfwSetDropCallback(window, drop);

    
    double aspect = Wdepth/Hdepth;
    double fov  = cam.fovy * 3.14159/180.0;
    double fy = (Hdepth/2.0) / tan( fov/2.0);
    double fx = (Wdepth/2.0) / tan( aspect * fov/2.0);

    printf("fx  = %lf\n", fx);
    printf("fy  = %lf\n", fy);

    // print version
    printf("MuJoCo Pro version %.2lf\n\n", mj_version());


    printf("number of position coordinates = %d\n", m->nq);
    printf("number of DOF = %d\n", m->nv);

    printf("time step = %lf\n ",m->opt.timestep);

    printf("camera focal lenght y:  [%lf ]\n", cam.fovy  );
    printf("z to the object:  [%lf ]\n", cam.distance);
    printf("cam azimuth:  [%lf ]\n", cam.azimuth);
    printf("cam elevation:  [%lf ]\n", cam.elevation);
    printf("cam lookat:  [%lf %lf %lf  ]\n", cam.lookat[0],cam.lookat[1],cam.lookat[2]);

    char szFile[32] = {0,};
    bool file_ready = false;
    int count = 0;
    while(!file_ready) {
       sprintf(szFile, "Log%03d.csv", count);
       if(0 != access(szFile, F_OK)) {
          file_ready=true;
       }
       if(count++ > 256) break;
    }
    if (file_ready) {
       image_dir="/tmp/";
       image_dir+=szFile;
       std::string c_dir = "mkdir -p "+image_dir+"/color/";
       std::string d_dir = "mkdir -p "+image_dir+"/depth/";
       system(c_dir.c_str());
       system(d_dir.c_str());
    }

    //  m->opt.timestep = 0.00001;
    // main loop
    std::vector<double> save_data;
    int rows=0;
    while( !glfwWindowShouldClose(window) ) {
        if (start) {
            static double data_start = data->time;
            save_data.push_back(data->time-data_start);
            rows++;
            for (int i=0; i<m->nq; i++) { // joint positions
                save_data.push_back(data->qpos[i]);
            }
            for (int i=0; i<m->nv; i++) { // velocities
                save_data.push_back(data->qvel[i]);
            }
            for (int i=0; i<m->nv; i++) { // accelerations
                save_data.push_back(data->qacc[i]);
            }
            for (int i=0; i<m->nu; i++) { // controls
                save_data.push_back(data->ctrl[i]);
            }

            // actuate
            if (data->ctrl[1] > -1.0) {
                data->ctrl[1]-= 0.05;
            }
            if (data->ctrl[2] > -1.0) {
                data->ctrl[2]-= 0.1;
            }
            mj_step(m,data);
        }

        render(window);

        // finalize
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    // File writing code 
    
    printf("nq: %d nv: %d nu: %d\n", m->nq, m->nv, m->nu);

    if (file_ready) {
        std::ofstream file_output;
        std::ofstream reported;
        std::ofstream contacts;
        file_output.open(szFile, std::ios::out);
        reported.open("reportedJointAngles.txt", std::ios::out);
        reported << rows <<std::endl;// rows
        reported << m->nq <<std::endl;// cols 
        contacts.open("reportedContacts.txt", std::ios::out);
        contacts << rows << std::endl;
        for (std::vector<double>::iterator it=save_data.begin(); it!=save_data.end(); ++it) {
            file_output << *it << ",";
            it++;
            // positions
            for(int id = 0; id < m->nq; id++) {
                file_output << *it << ",";
                reported << *it << " ";
                it++;
            }
            reported << std::endl;
            contacts << "0" << std::endl;
            // velocities
            for(int id = 0; id < m->nv; id++) {
                file_output << *it << ",";
                it++;
            }
            // acclerations
            for(int id = 0; id < m->nv; id++) {
                file_output << *it << ",";
                it++;
            }
            // controls
            for(int id = 0; id < m->nu; id++) {
                file_output << *it << ",";
                it++;
            }
            it--;
			file_output << std::endl;
        }
    }
    std::string copy_cmd="cp ";
    copy_cmd=copy_cmd+szFile+" "+image_dir+"/";
    system(copy_cmd.c_str());

    std::string copy_rpt="cp reportedJointAngles.txt ";
    copy_rpt=copy_rpt+image_dir+"/";
    system(copy_rpt.c_str());

    std::string copy_ct="cp reportedContacts.txt ";
    copy_ct=copy_ct+image_dir+"/";
    system(copy_ct.c_str());

    // delete everything we allocated
    mj_deleteData(data);
    mj_deleteModel(m);
    mjr_freeContext(&con);
    mjv_freeObjects(&objects);

    // terminate
    glfwTerminate();
    mj_deactivate();
    return 0;
}
