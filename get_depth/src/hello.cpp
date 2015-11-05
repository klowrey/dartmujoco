#include "mujoco.h"
#include "stdio.h"

char error[1000];
mjModel* m;
mjData* d;

int main(void)
{
    
     mj_activate("../key/mjkey.txt");
   // load model and check for errors
   m = mj_loadXML("../models/box.xml", error);
   if( !m )
   {
      printf("%s\n", error);
      return -1;
   }

   // make data corresponding to model
   d = mj_makeData(m);

   // run simulation for 10 seconds
   while( d->time<10 )
      mj_step(m, d);

   // free model and data
   mj_deleteData(d);
   mj_deleteModel(m);
   return 0;
}