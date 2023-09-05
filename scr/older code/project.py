import os
import copy
import statistics
from pathlib import Path
from time import perf_counter
import geomie3d

#=================================================================================================================================
# region: PARAMETERS
#=================================================================================================================================
high_res_dir = "/princeton mrt sensor\\sample ply data\\ThermalArray_08-31-2023_11-01-13_0.PLY"
room_size = [10,10,10] # xsize ysize zsize
sensor_pos = [0, 0, 1.5] # xpos ypos zpos
viz = True
# endregion: PARAMETERS
#=================================================================================================================================
# region: FUNCTIONS
#=================================================================================================================================
def write2ply(res_path, xyz_ls, temp_ls, header):
    nvs = len(xyz_ls)
    nvs_line = 'element vertex ' + str(nvs) + '\n'
    v_cnt = 0
    for cnt, h in enumerate(header):
        hsplit = h.split(' ')
        if hsplit[0] == 'element':
            if hsplit[1] == 'vertex':
                v_cnt = cnt
    
    header_w = header[:]
    header_w[v_cnt] = nvs_line
    for cnt,xyz in enumerate(xyz_ls):
        temp = temp_ls[cnt]
        v_str = str(xyz[0]) + ' ' + str(xyz[1]) + ' ' + str(xyz[2]) + ' ' + str(temp) + '\n'
        header_w.append(v_str)
    
    f = open(res_path, "w")
    f.writelines(header_w)
    f.close()
    
def check_ply_file(header):
    ref_h = ['ply', 'format ascii 1.0',
             'element vertex',
             'property float x', 
             'property float y', 
             'property float z', 
             'property float temperature', 
             'end_header']
    
    check = 0
    for h in header:
        h = h.lower()
        h = h.replace('\n','')
        
        hsplit = h.split(' ')
        if hsplit[0] == 'comment':
            ctype = hsplit[1]
            if ctype=='date' or ctype=='time' or ctype=='sensorid' or ctype=='sensortype':
                h = hsplit[0] + ' ' + ctype
        
        elif hsplit[0] == 'element':
            if hsplit[1] == 'vertex':
                h = hsplit[0] + ' ' + hsplit[1]
                
        if h in ref_h:
            # print(h)
            check+=1
    
    if check == 8:
        return True
    else:
        return False
    
def read_therm_arr_ply(file_path):    
    with open(file_path) as f:
        lines = f.readlines()
    nheaders = 8
    #check if this is a valid chaosense file
    headers = lines[0:nheaders]
    isValid = check_ply_file(headers)
    if isValid:
        xyzs = []
        verts_data = lines[nheaders:]
        temp_ls = []
        temp_dls = []
        for v in verts_data:
            v = v.replace('\n', '')
            v = v.replace('\t', ' ')
            vsplit = v.split(' ')
            vsplit = list(map(float, vsplit))
            xyzs.append(vsplit[0:3])
            temp_dls.append({'temperature':vsplit[3]})
            temp_ls.append(vsplit[3])
        
        v_ls = geomie3d.create.vertex_list(xyzs, attributes_list=temp_dls)
        return v_ls, temp_ls, headers
    else:
        return [], [], headers
    
def project(therm_scan_path, sensor_pos, scene_faces, viz = False):
    verts_data, temp_ls, headers = read_therm_arr_ply(therm_scan_path)
    cmp = geomie3d.create.composite(verts_data)
    # axis = [0,0,1]
    # rotation = 0
    # cmp = geomie3d.modify.rotate_topo(cmp, axis, rotation)
    verts_data = geomie3d.get.vertices_frm_composite(cmp)
    #convert the verts to rays and 
    rays = []
    for vcnt,v in enumerate(verts_data):
        temp = v.attributes['temperature']
        ray = geomie3d.create.ray(sensor_pos, v.point.xyz, attributes = {'temperature':temp, 'id': vcnt})
        rays.append(ray)
        
    hrays,mrays,hit_faces,miss_faces = geomie3d.calculate.rays_faces_intersection(rays, scene_faces)
    if viz == True:
        geomie3d.viz.viz_falsecolour(verts_data, temp_ls)
        vcmp = geomie3d.create.composite(verts_data)
        mv_v = geomie3d.modify.move_topo(vcmp, sensor_pos)
        mv_vs = geomie3d.get.vertices_frm_composite(mv_v)
        
        cmp = geomie3d.create.composite(scene_faces)
        edges = geomie3d.get.edges_frm_composite(cmp)
        geomie3d.viz.viz_falsecolour(mv_vs, temp_ls, other_topo_dlist=[{'topo_list':edges, 'colour':'white'}])
        viz_projection(hrays, scene_faces)
    return hrays,mrays,hit_faces,miss_faces, headers

def calc_avg_temp_srf(proj_face):
    att = proj_face.attributes
    if 'rays_faces_intersection' in att:
        int_att = att['rays_faces_intersection']
        rays = int_att['ray']
        if len(rays) !=0:
            temp_ls = [r.attributes['temperature'] for r in rays]
            avg_temp = statistics.median(temp_ls)
            geomie3d.modify.update_topo_att(proj_face, {'temperature':avg_temp})
            return avg_temp
        else:
            return None

def viz_projection(hrays, scene_faces):
    v_ls = []
    temp_ls = []
    for hray in hrays:
        att = hray.attributes
        att2 = att['rays_faces_intersection']
        intersects = att2['intersection']
        temp = att['temperature']
        for intersect in intersects:
            v = geomie3d.create.vertex(intersect, attributes={'temperature':temp})
            temp_ls.append(temp)
            v_ls.append(v)
    
    cmp = geomie3d.create.composite(scene_faces)
    edges = geomie3d.get.edges_frm_composite(cmp)
    geomie3d.viz.viz_falsecolour(v_ls, temp_ls, 
                                     other_topo_dlist=[{'topo_list': edges, 'colour': 'white'}], 
                                     false_min_max_val=None)
# endregion: FUNCTIONS
#=================================================================================================================================
# region: MAIN
#=================================================================================================================================
# create a box with geomie3d library
t0 = perf_counter()
box = geomie3d.create.box(10, 10, 10)
box = geomie3d.modify.move_topo(box, (0,0,5), (0,0,0))
face_list = geomie3d.get.faces_frm_solid(box)
rev_faces = []
for cnt,f in enumerate(face_list):
    rev_f = geomie3d.modify.reverse_face_normal(f)
    geomie3d.modify.update_topo_att(rev_f, {'name': 'surface' + str(cnt)})
    rev_faces.append(rev_f)

# if viz == True:
#     geomie3d.viz.viz([{'topo_list':[box], 'colour':'red'}])

#read the file and get all the vert data
parent_path = Path(high_res_dir).parent.absolute()
res_dir = os.path.join(parent_path, 'projected')
if not os.path.isdir(res_dir): 
    os.mkdir(res_dir)
    
ff_ls = sorted(os.listdir(high_res_dir))
ply_name_ls = []
for d in ff_ls:
    file_ext = d.split('.')[-1].lower()
    if file_ext == 'ply':
        ply_name_ls.append(d)

t1 = perf_counter()
print('Time taken to process base scene (mins):', round((t1-t0)/60, 1))

for cnt, ply_name in enumerate(ply_name_ls):
    new_bdry_srfs = copy.deepcopy(rev_faces)
    t2 = perf_counter()
    ply_path = os.path.join(high_res_dir, ply_name)
    hrays,mrays,hit_faces,miss_faces,header = project(ply_path, sensor_pos, new_bdry_srfs, viz=viz)
    xyz_ls = []
    temp_ls = []
    for hray in hrays:
        att = hray.attributes
        int_att = att['rays_faces_intersection']
        xyz = int_att['intersection'][0]
        fs = int_att['hit_face']
        temp = att['temperature']
        xyz_ls.append(xyz)
        temp_ls.append(temp)
    
    avg_temps = []
    for hf in hit_faces:
        avg_temp = calc_avg_temp_srf(hf)
        avg_temps.append(avg_temp)
        att = {'name': hf.attributes['name'], 'temperature': avg_temp}
        geomie3d.modify.overwrite_topo_att(hf, att)
    
    if viz == True:
        geomie3d.viz.viz_falsecolour(hit_faces, avg_temps)

    # print(ttl_rays_cnt)
    nsplit = ply_name.split('.')
    res_path1 = os.path.join(res_dir, nsplit[0] + '_projected.ply' )
    res_path2 = os.path.join(res_dir, nsplit[0] + '_projected.geomie3d' )
    geomie3d.utility.write2geomie3d(hit_faces, res_path2)
    write2ply(res_path1, xyz_ls, temp_ls, header)
    t3 = perf_counter()
    time_take = round((t3-t2)/60,1)
    print('Time taken to project one file (mins):', time_take, cnt)

# endregion