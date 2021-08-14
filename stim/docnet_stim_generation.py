import bpy 
import random
import numpy as np
import os
from glob import glob
from math import radians
import time
import pdb
import sys

#terminal commands: 
#/lab_data/hawk/blender/blender-2.93.2/blender -b docnet_image_creation.blend -P docnet_stim_generation.py
bpy.context.scene.render.engine = 'CYCLES'
curr_dir = '/home/vayzenbe/GitHub_Repos/docnet/stim'
os.chdir(curr_dir)
model_dir = '/lab_data/behrmannlab/image_sets/ShapeNetCore.v2'
out_dir= '/lab_data/behrmannlab/image_sets/ShapeNet_images'
#im_dir = f'{curr_dir}/obj_images'

#Start orient
angle_increments = 30
min_angle = (-75)
max_angle = 75
#num_orients = max_angle/angle_increments

#set background color
#bpy.context.scene.world.horizon_color = (.184, .184, .184)

#how many objects to use for each class
num_obj = 2

#Load model list
#model_list = np.loadtxt(f'{curr_dir}/model_list.csv', delimiter=',', dtype=object)



def create_object(obn, ob):
    obj_name = ob.split('/')[-1]

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
    rand_rot = random.randint(min_angle,max_angle)
    #db.set_trace()

    bpy.context.object.rotation_euler.z = radians(rand_rot)
    #bpy.context.object.rotation_euler.z = radians(random.randint(-65,65))

    #Set file path for the render
    #pdb.set_trace()
    bpy.context.scene.render.filepath = f'{out_dir}/{cl.split("/")[-1]}/{obj_name}.jpg'

    #Take the picture
    bpy.ops.render.render(write_still = True)



    #Delete selected object
    bpy.ops.object.delete()

#load class name from python
cl = sys.argv[1]

#create image directory for object class
os.makedirs(f'{out_dir}/{cl.split("/")[-1]}', exist_ok = True)

#load all folders in class folder
exemplar_list = glob(f'{cl}/*')

#shuffle exemplar list
random.shuffle(exemplar_list)

#loop through objects in class folder
for obn, ob in enumerate(exemplar_list[:num_obj]):  
    result = create_object(obn, ob)

#result.compute()

print('')
print('')
print('***TOTAL ELAPSED TIME***', time.perf_counter()-t1)
