import bpy 
import random
import numpy as np
import os
from glob import glob
from math import radians

#blender loc: /lab_data/hawk/blender/blender-2.93.2
bpy.context.scene.render.engine = 'CYCLES'
curr_dir = '/home/vayzenbe/GitHub_Repos/docnet/stim'
os.chdir(curr_dir)
model_dir = '/lab_data/behrmannlab/image_sets/ShapeNetCore.v2'
out_dir= '/lab_data/behrmannlab/image_sets/ShapeNet_images'
im_dir = f'{curr_dir}/obj_images'

#Start orient
angle_increments = 30
min_angle = -90
max_angle = 90
#num_orients = max_angle/angle_increments

#set background color
#bpy.context.scene.world.horizon_color = (.184, .184, .184)

#how many objects to use for each class
num_obj = 1

#Load model list
#model_list = np.loadtxt(f'{curr_dir}/model_list.csv', delimiter=',', dtype=object)

cat_folders = glob(f'{model_dir}/*')

#loop through object classes
for cln, cl in enumerate(cat_folders):
    exemplar_list = glob(f'{cl}/*')



    if len(exemplar_list) > 300:

        #create image directory for object class
        os.makedirs(f'{out_dir}/{cl[-8:]}', exist_ok = True)
        
        #load all folders in class folder
        
        
        #shuffle exemplar list
        random.shuffle(exemplar_list)
        
        #loop through objects in class folder
        for obn, ob in enumerate(exemplar_list[:num_obj]):    
            #unselect everything
            bpy.ops.object.select_all(action='DESELECT')
            
            #import object model
            imported_object = bpy.ops.import_scene.obj(filepath=f'{ob}/models/model_normalized.obj')
            print('object loaded')

            '''
            #Set file path for the render
            bpy.context.scene.render.filepath = f'{curr_dir}/obj_images/test.jpg'

            #Take the picture
            bpy.ops.render.render(write_still = True)
            '''
            #set current object to variable 
            curr_obj = bpy.context.selected_objects[0]

            #select/activate it
            curr_obj.select_set(True)
            bpy.context.view_layer.objects.active = curr_obj
            
            #change scale if relevant
            #bpy.context.object.scale = [float(model_list[cln,2]), float(model_list[cln,2]), float(model_list[cln,2])]
            #bpy.context.object.location.z =  float(model_list[cln,3])
            
            
            #Remove the material from the object
            for mat_n, material in enumerate(bpy.data.materials):
                # clear and remove the old material
                #print(material)
                material.user_clear()
                bpy.data.materials.remove(material)
                #print(mat_n)
            
                
            ob = bpy.context.active_object
            for mn in range(0, len(ob.data.materials)):
                
                bpy.context.object.active_material_index = mn
                mat = bpy.data.materials.new(name=f"Material_{mn}")
                mat.diffuse_color = (.5, 0, 0,1) 
                
                #try:
                # assign to 1st material slot
                
                ob.data.materials[mn] = mat
                
                #ob.data.materials.append(mat)

            #break
                #except:
                    #continue
      
                #bpy.context.object.active_material.use_nodes = True
                bpy.data.materials[f"Material_{mn}"].specular_color = (0, 1, 0.5)

             #   #select that index and add a new material
              #  bpy.context.object.active_material_index = mn
               # bpy.ops.material.new()
                
                #rotate object
            bpy.context.object.rotation_euler.z = radians(random.randint(-90,90))
        
            #Set file path for the render
            bpy.context.scene.render.filepath = f'{curr_dir}/obj_images/{model_list[cln,1]}/{model_list[cln,1]}_{obn}_{curr_orient}.jpg'

            #Take the picture
            bpy.ops.render.render(write_still = True)
        
        
        
            #Delete selected object
            bpy.ops.object.delete()
            
