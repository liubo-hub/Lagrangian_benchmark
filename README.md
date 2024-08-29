# **Lagrangian Particle Fluid Benchmark Open Dataset**
 
![](https://ai-studio-static-online.cdn.bcebos.com/433a5637354940f48f383b6e4d356db0ad0a976f1d4c4bc8b1d45b52f4d5c702)
* **Lagrangian particle methods for PDE solvers have natural advantages in handling free surface or complex fluid-solid interactions.**

* **Establishing a comprehensive and diverse public benchmark dataset will greatly benefit the comparison of different scientific AI models and the improvement of existing models. This project uses Taichi open-source SPH, MPM, and DEM particle solvers to generate datasets for granular flows, dam breaks, Taylor-Green vortices, lid-driven cavity flows, jets, and more for training and evaluating advanced AI learners.**

[Project Source: AI Studio (Co Creation Plan Competition)](https://aistudio.baidu.com/projectdetail/7507477)
# 1 Introduction to Methods
 
## **1.1 SPH**
 
![](https://ai-studio-static-online.cdn.bcebos.com/c16478b148234dc98efea6c2862e0d2c8f13b26b265d466495b2d190a919332b)

* **SPH（Smoothed Particle Hydrodynamics）**
is a particle-based numerical method primarily used for simulating fluid dynamics. Initially developed to simulate galaxy motion in astrophysics, it is now widely used in computational fluid dynamics, structural mechanics, and other fields.

* **Basic Principle:**
Based on the Lagrangian approach, it divides the fluid or material into small particles and uses a smoothing kernel function to interpolate the motion of these particles, mainly for fluid dynamics simulation.

* **Applications: **
It is primarily used in fluid dynamics simulations, such as hydrodynamics, oceanography, meteorology, etc. 

The above figure shows the simulation of SPH incompressible fluid, cited from Mackin M, Muller M. Position based fluids. ACM Trans.on Graphics, 2013, 32 (4): 104. [doi. 10.1145/2461912.2461984]
 
## **1.2 DEM**
 
![](https://ai-studio-static-online.cdn.bcebos.com/3028fbe80b0043c5b235dc1f3f821da97501695246cc4d2aabfc3422e9bc69e2)

* **DEM（Discrete Element Method）**
 is a numerical method used to simulate the mechanical behavior of granular materials. It treats granular materials as collections of many discrete particles, and by simulating the interactions between these particles, it can study the deformation, flow, and collision behavior of granular materials under different conditions.
* **Basic Principle:**
 It divides objects into discrete particles or rigid bodies, simulating their interactions, mainly for problems involving granular materials, granular flow, and particle collisions.
* **Applications: **
Suitable for simulating the behavior of granular materials, such as granular flow, vibratory sieving, and particle collisions, widely used in particle technology and geotechnical engineering.

The above figure shows the numerical simulation of an industrial fluidized bed (CFB) reactor using Fluent (DDPM+DEM)
## **1.3 MPM**
 
![](https://ai-studio-static-online.cdn.bcebos.com/e6f6ff42a2ba4d0dbab86b5723a93fd8f914d773a1b74114aa8c9809623fb5a1)

* **MPM（Material Point Method）**
is a numerical method used for simulating material deformation and fluid-solid interactions. It is a hybrid method combining characteristics of Lagrangian and Eulerian approaches, primarily used for handling large deformations, multiphase flows, material breakage, and more.
* **Basic Principle:**：
It represents materials as a collection of particles and tracks their movement on an Eulerian grid, combining the advantages of Lagrangian and Eulerian methods. It is mainly used for simulating the interaction between solids and fluids, suitable for large deformation and multiphase flow problems.
* **Applications:**
Mainly used for simulating the deformation, breakage, and flow of solid materials, applicable in civil engineering, geology, materials science, etc.

The above picture is quoted from Disney's "Frozen Production Process Disclosure"

# 2 Method Comparison
 
**SPH, DEM, and MPM are numerical methods used for simulating material behavior, each with its own advantages and disadvantages:**
* **SPH（Smoothed Particle Hydrodynamics）:**
* Advantages: Suitable for free surface fluid dynamics problems, such as water flow, droplets, etc.; particle interactions can be naturally simulated; suitable for handling large deformations and breakage problems.
* Disadvantages: Boundary conditions are relatively complex to handle, requiring special techniques; the number of particles increases with the complexity of the problem, leading to higher computational costs; not well-suited for solid material behavior.
* **DEM（Discrete Element Method）:**
* Advantages: Suitable for handling granular flows, granular-structure interactions, etc.; accurately models collision and contact behavior between particles; relatively easy to handle boundary conditions.
* Disadvantages: Difficult to handle fluid-particle interactions, unable to simulate fluid behavior; computational costs increase significantly with the number of particles; not suitable for handling large deformations and breakage problems.

* **MPM（Material Point Method）:**
* Advantages: Combines the advantages of particle methods and grid methods, suitable for multiphase and multi-material problems; effectively handles large deformations and breakage problems; suitable for coupling with reinforcement learning and other machine learning methods.
* Disadvantages: May require additional techniques when dealing with free surface fluid dynamics problems; for some problems, the fixed structure of the grid may result in lower computational efficiency compared to other methods; DEM may be more suitable for specific problems such as granular flow.

Overall, the choice of numerical method depends on the nature of the problem being studied. In practice, researchers may choose the appropriate method based on the characteristics of the problem, or even combine different methods to fully leverage their advantages. 
# 3 Program Execution

**Install**

```
pip install -r requirements.txt
```
**Train on the Lagrangebench Dataset**
```
python lagrangebench-main/main.py --load_ckp=ckp/gns_dam2d_20240227-133120/best rollout_dir=rollout/gns_dam2d_20240227-133120/best mode=infer test=True
```

**Continue Training**
```
python --load_ckp=ckp/gns_dam2d_20240227-133120/best
```
 

**2D Rendering:**
Rendered GIFs are saved in gif.

```
python renderer.py
```
 
**3D Rendering:**
  
```
python renderer_3D.py
```
 
**MPM Example**

```
python MPM_Taichi/run_mpm.py --input_path=MPM_Taichi/examples/slope_sand/inputs_2d_gns.json
```
 
![](https://ai-studio-static-online.cdn.bcebos.com/fbab9e3129b946cfb1b46e284a662fb005792bc8704f4d948054c25800a78315)

 
**DEM Example**

![](https://ai-studio-static-online.cdn.bcebos.com/5c09cfa69b334a4583c54d41387c7640f7d1fd5397e24b44b6b6f97b06fe83b7)
![](https://ai-studio-static-online.cdn.bcebos.com/489eb25cdc404ed9aa021a347d5d93135bf390847cac4e03a50172dddc886d81)

 
# 4 Dataset Description

[Partial dataset download address](https://aistudio.baidu.com/projectdetail/7507477)

**1.Structure：**

**The ".h5" format contains a variable-length list of tuples, where each tuple corresponds to a different training trajectory. The format is (position information, particle type information), where the position information of particles is a three-dimensional tensor with the shape (n_time_steps, n_particles, n_dimensions). The particle type is a one-dimensional tensor with the shape (n_particles), where the values from 0 to 9 represent different types of particles.**

**Example with test.h5: test["trajectory_01"][0][0][0][0]**
* trajectory_01 represents a specific scene, including all particle information in that scene.
* The first [0] represents the particle position information for all frames.
* The second [0] represents the position information of all particles in the first frame of the scene.
* The third [0] represents the information of the first particle in the first frame [x, y, z].
* The fourth [0] represents the x-coordinate information of the first particle.

**2.Application Scenarios:**
* 2D_SAND: Can be used to simulate dam breaks and study the collapse of sand under different attributes (such as friction coefficient and restitution coefficient).
* 3D_RPF: Reverse Poiseuille Flow, which can deepen our understanding of fluid behavior, especially under non-traditional conditions or in non-Newtonian fluids.
* 3D_DEM: Can be used to study the motion of particles with different attributes. 
* slope_dem and slope_mpm: Used to study the motion of granular fluids under the influence of barriers. 
* slope_two_component_dem: Used to study the interaction of different types of particles with multiple components (different densities, masses, volumes) and the mixing of particles under different components. 
* .........

**MPM generation (gns training inference):**
 
![](https://ai-studio-static-online.cdn.bcebos.com/d01e99165f6746f785b307a49efa5f13545571962e0046e58a1615976ba7f651)

 
**DEM generation (gns training inference):**
 
![](https://ai-studio-static-online.cdn.bcebos.com/fcbd8f4d3e9e47bc8327521ddd43ff6cb8d54a0707df4c058c5c8651b0765f15)

 
**DEM multi-component particles (yellow particles have three times the density of blue particles):**
 
![](https://ai-studio-static-online.cdn.bcebos.com/844d3bf1f81548938344af4a29a27d976a8ca75c8bbb44c4bff6326e4836b62e)

**DEM 3D baffle (training inference):**
 
![](https://ai-studio-static-online.cdn.bcebos.com/5273aa2fd92a4e4fa1dce208417b8138a8422883ab3147ef91e822a5a9f66bd7)

 
![](https://ai-studio-static-online.cdn.bcebos.com/f0a0181b4e5544d8a6825bdd980f734ec374e4e92e5c4fa7b3e144b14383e346)

 
![](https://ai-studio-static-online.cdn.bcebos.com/30576ebe2ccd467691e4c482de677bd4d858e8d4b7e741c8bc4b6507497b10ef)

 
![](https://ai-studio-static-online.cdn.bcebos.com/13ab5e890a8045819ec10130af0a3c049c28645d6d2a450a8f44e50184a986f0)

 
![](https://ai-studio-static-online.cdn.bcebos.com/d0ebbdc80a584e77aa60d85746be65cd176f6e6bb0e2424bab9aecbd28de522c)

 
# 5 Schematic diagram of dataset generation
 
### **1. Flowchart for generating sph and mpm datasets:**
![](https://ai-studio-static-online.cdn.bcebos.com/c8395bf6e35e42bf8fa7785a80d8808c4cbb8420ec47498aaa872e9dd7b79479)

 
### **2. DEM dataset flowchart:**
![](https://ai-studio-static-online.cdn.bcebos.com/5f77ec65434543e9a39cf8bfa04f2fe468805c12636f4d8f9526dd2cfe4f8d3e)


## Reference project address:
 
**1. Training network source project address:**[https://github.com/tumaer/lagrangebench/tree/main](http://)

**2. SPH_Taichi Source Project Address:**[https://github.com/erizmr/SPH_Taichi](http://)

**3. MPM_Taichi Source Project Address:**[https://github.com/yjchoi1/taichi_mpm](http://)

**4. Mfix software address:**[https://mfix.netl.doe.gov/products/mfix/](http://)
