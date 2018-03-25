/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	
	default_random_engine gen;
	if (!is_initialized){
		normal_distribution<double> dist_x(x, std[0]);
		normal_distribution<double> dist_y(y, std[1]);
		normal_distribution<double> dist_theta(theta, std[2]);

		num_particles = 100; 
		
		Particle p;

		for(int i = 0; i < num_particles; i++){
			p.id = i;
			p.x = dist_x(gen);
			p.y = dist_y(gen);
			p.theta = dist_theta(gen);
			p.weight = 1.0;
			particles.push_back(p);
			weights.push_back(1.0);
		}
		is_initialized = true;
	}

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;

	// creating zero mean distributions in order to use as additive noise 
	normal_distribution<double> dist_x(0.0, std_pos[0]);
	normal_distribution<double> dist_y(0.0, std_pos[1]);
	normal_distribution<double> dist_theta(0.0, std_pos[2]);

	if (yaw_rate == 0){
		double vel_integrated = velocity * delta_t; 
		for(int i=0; i < num_particles; i++){
			particles[i].x += vel_integrated * cos(particles[i].theta) + dist_x(gen);
			particles[i].y += vel_integrated * sin(particles[i].theta) + dist_y(gen);
			particles[i].theta += dist_theta(gen);
		}
	}
	else{
		double v_over_thetadot = velocity / yaw_rate;
		double thetadot_times_dt = yaw_rate * delta_t;
		
		for(int i=0; i < num_particles; i++){
			double theta_next = particles[i].theta + thetadot_times_dt;
			particles[i].x += v_over_thetadot * (sin(theta_next) - sin(particles[i].theta)) + dist_x(gen);
			particles[i].y += v_over_thetadot * (cos(particles[i].theta) - cos(theta_next)) + dist_y(gen);
			particles[i].theta = theta_next + dist_theta(gen);
		}
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	// Not used! Associations handled in updateWeights.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	

	// Note on source: a nice summary of procedure to be followed in update weights is 
	// given in the Udacity forum thread  : https://discussions.udacity.com/t/c-help-with-dataassociation-method/291220/5
	// Below code highly influenced by the post by user "driveWell"

	//constants for weight calculation:
	const double gauss_norm= 1.0 / (2.0 * M_PI * std_landmark[0] * std_landmark[1]);
	const double sigma_x_term = -0.5 / (std_landmark[0] * std_landmark[0]);
	const double sigma_y_term = -0.5 / (std_landmark[1] * std_landmark[1]);
	//cout<<"update";
	// Iterating through each particle
	for(int i=0; i < particles.size(); i++){
		double cos_theta = cos(particles[i].theta);
		double sin_theta = sin(particles[i].theta);

		// initializing particle weight to 1.0, we will accumulate probability with each observation:
		particles[i].weight = 1.0;

		// check each obseration and transform into map coordinates
		vector<double> feature_distances(map_landmarks.landmark_list.size());
		for(int obs_ind=0; obs_ind < observations.size(); obs_ind++){
			LandmarkObs obs_t;
			feature_distances.clear();
			obs_t.id = observations[obs_ind].id;
			obs_t.x = observations[obs_ind].x * cos_theta - observations[obs_ind].y * sin_theta + particles[i].x;
			obs_t.y = observations[obs_ind].x * sin_theta + observations[obs_ind].y * cos_theta + particles[i].y;

			for(Map::single_landmark_s feature : map_landmarks.landmark_list){
				// is map feature in line of sight?
				double f_dist = dist(particles[i].x, particles[i].y, feature.x_f, feature.y_f);

				if (f_dist <= sensor_range){
					// add the distance between the observation and landmark to the distances list. 
					feature_distances.push_back(dist(obs_t.x, obs_t.y, feature.x_f, feature.y_f));
				}
				else{
					// feature cannot be observed from this particle. assign "infinite" distance.
					feature_distances.push_back(99999.0); 	
				}	
			}

			// we are still looping on observations. incorparating current observation's probability into particle weight:

			// find closest feature to the observation using std::min_element and std::distance (http://en.cppreference.com/w/cpp/algorithm/min_element)
			vector<double>::iterator iterator_min = min_element(begin(feature_distances), end(feature_distances)); 
	
			int index_min = distance(begin(feature_distances), iterator_min);
	
			double x_nearest_f = map_landmarks.landmark_list[index_min].x_f;
			double y_nearest_f = map_landmarks.landmark_list[index_min].y_f; 
	
			//probability of predicted observation matching to the closest feature on map:
			double match_prob = gauss_norm * exp(sigma_x_term * (obs_t.x - x_nearest_f) * (obs_t.x - x_nearest_f)
												+ sigma_y_term * (obs_t.y - y_nearest_f) * (obs_t.y - y_nearest_f));

			particles[i].weight *= match_prob; // incorparate into overall particle weight
		}
		weights[i] = particles[i].weight; // add the weight of current particle to weights vector, to be used in resampling. 
	}

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	// Heavily influenced by Project Q&A video.. (https://www.youtube.com/watch?v=-3HI3Iw3Z9g&index=3&list=PLAwxTw4SYaPnfR7TzRZN-uxlxGbqxhtm2)
	default_random_engine gen;
	discrete_distribution<int> distribution(weights.begin(), weights.end());
	vector<Particle> resampled_particles;

	for(int i=0; i< particles.size(); i++){
		resampled_particles.push_back(particles[distribution(gen)]);
	}

	particles = resampled_particles;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapM_PIng to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapM_PIng already converted to world coordinates
    // sense_y: the associations y mapM_PIng already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
