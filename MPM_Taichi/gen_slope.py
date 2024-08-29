import random

import numpy as np
import taichi as ti
# ti.init(arch=ti.cuda, device_memory_fraction=0.9)
# ti.init(arch=ti.cuda, device_memory_GB=4)
ti.init(arch=ti.cpu, cpu_max_num_threads=16)
combined_particles = ti.Vector.field(2, dtype=ti.f32, shape=800)

def load_and_copy_data(particles_data):
    data = gen_slope_particles()
    for i in range(len(data)):
        particles_data[i] = data[i]
def generate_symmetric_line_particles(num_particles, x_range, y_range):
    x_values = np.linspace(x_range[0], x_range[1], num_particles)
    y_values = np.linspace(y_range[0], y_range[1], num_particles)
    particles = np.column_stack((x_values, y_values))

    # Generate symmetric particles
    symmetric_particles = np.copy(particles)
    symmetric_particles[:, 0] = 1.0 - particles[:, 0]

    return particles, symmetric_particles

def rotate_particles(particles, angle_change_degrees):
    bottom_particle_index = np.argmin(particles[:, 1])
    bottom_particle = particles[bottom_particle_index]
    angle_change_radians = np.radians(angle_change_degrees)
    rotated_particles = np.copy(particles)
    rotated_particles[:, 0] -= bottom_particle[0]
    rotated_particles[:, 1] -= bottom_particle[1]
    rotation_matrix = np.array([[np.cos(angle_change_radians), -np.sin(angle_change_radians)],
                                [np.sin(angle_change_radians), np.cos(angle_change_radians)]])
    rotated_particles[:, :2] = np.dot(rotated_particles[:, :2], rotation_matrix.T)
    rotated_particles[:, 0] += bottom_particle[0]
    rotated_particles[:, 1] += bottom_particle[1]
    return rotated_particles

def gen_slope_particles():
    # Generate 100 particles forming a symmetric line
    num_particles = 400
    x_range = [0.56, random.uniform(0.79, 0.95)]
    y_range = [0.4, 0.7]
    particles, symmetric_particles = generate_symmetric_line_particles(num_particles, x_range, y_range)

    combined_particles = np.concatenate((particles, symmetric_particles), axis=0)

    return combined_particles

# def run():
#     np_particles = gen_slope_particles()
#     copy_np_to_ti(np_particles)

# # Rotate original symmetric particles by 45 degrees while keeping the bottom particle fixed
# rotated_particles = rotate_particles(particles, 5.0)
#
# # Rotate symmetric particles by -45 degrees while keeping the bottom particle fixed
# rotated_symmetric_particles = rotate_particles(symmetric_particles, -5.0)


