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
	num_particles = 100;
	
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
	
	for (int i=0; i<num_particles; ++i) {
		Particle sample_particle;
		sample_particle.id = float(i);
		sample_particle.x = dist_x(gen);
		sample_particle.y = dist_y(gen);
		sample_particle.theta = dist_theta(gen);
		sample_particle.weight = 1.0f;
		
		particles.push_back(sample_particle);
	}
	
	weights.resize(num_particles);
	
	is_initialized = true;
	

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	
	default_random_engine gen;
	
	//create noise
	normal_distribution<double> noise_x(0.0, std_pos[0]);
	normal_distribution<double> noise_y(0.0, std_pos[1]);
	normal_distribution<double> noise_theta(0.0, std_pos[2]);

	//predict
	for (int i=0; i<particles.size(); ++i) {
		Particle& particle = particles[i];
		
		if (fabs(yaw_rate) < 0.00001) {
			particle.x += velocity * delta_t * cos(particle.theta);
			particle.y += velocity * delta_t * sin(particle.theta);
		} else {
			particle.x += (velocity / yaw_rate) * (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta)) + noise_x(gen);
			particle.y += (velocity / yaw_rate) * (cos(particle.theta) - cos(particle.theta + yaw_rate * delta_t)) + noise_y(gen);
			particle.theta += yaw_rate * delta_t + noise_theta(gen);
		}
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (int i=0; i < observations.size(); ++i) {

	 	LandmarkObs obs = observations[i];

	 	int closest = -1;

	 	//preset min distance
	 	double distance_min = 10000.0;
		
	 	// for each prediction
	 	for (int j=0; j < predicted.size(); ++j) {

	 		// calculate distance between predicted measurement and obs
	 		double distance = dist(predicted[j].x, predicted[j].y, obs.x, obs.y);

	 		if (distance < distance_min) {
	 			distance_min = distance;
	 			closest = predicted[j].id;
	 		}
	 	}

	 	observations[i].id = closest;
		
	 }

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
	
	double s_x = std_landmark[0];
	double s_y = std_landmark[1];
	double gauss_norm = 1.0 / (s_x*s_y*2.0*M_PI);
	
	for (int i=0; i<num_particles; ++i) {
		Particle& p = particles[i];
		
		p.weight = 1.0;
		
		vector<LandmarkObs> predictions;
		
		//use landmark only in sensor range
		for (int j=0; j<map_landmarks.landmark_list.size(); ++j) {
			Map::single_landmark_s lm = map_landmarks.landmark_list[j];
			double lm_x = lm.x_f;
			double lm_y = lm.y_f;
			int lm_id = lm.id_i;
			
			if (fabs(lm_x - p.x) <= sensor_range && fabs(lm_y - p.y) < sensor_range) {
				predictions.push_back(LandmarkObs{lm_id, lm_x, lm_y});
			}
		}
		
		long double w = 1.0;
		
		for (int k=0; k<observations.size(); ++k) {
			LandmarkObs obs = observations[k];
			//transform conditions
			double obs_map_x = p.x + obs.x * cos(p.theta) - obs.y * sin(p.theta);
			double obs_map_y = p.y + obs.x * sin(p.theta) + obs.y * cos(p.theta);
			
			//do data association
			double dist_min = 10000.0;
			double pred_x_min, pred_y_min;
			
			for (int j=0; j<predictions.size(); ++j) {
				LandmarkObs pred = predictions[j];
				double distance = dist(pred.x, pred.y, obs_map_x, obs_map_y);
				
				if (distance < dist_min) {
					dist_min = distance;
					pred_x_min = pred.x;
					pred_y_min = pred.y;
				}
					
				
			}
			
			double p_weight = gauss_norm * exp( -(0.5*pow( (obs_map_x-pred_x_min)/s_x, 2.0 )+0.5*pow( (obs_map_y-pred_y_min)/s_y, 2.0 ) ));
			
			w *= p_weight;
		}
		
		p.weight = w;
		weights[i] = w;
		
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	
	default_random_engine gen;
	
	discrete_distribution<> dist_weighted(weights.begin(), weights.end());
	vector<Particle> particles_sampled;
	
	for (int i=0; i<particles.size(); ++i) {
		int sample_index = dist_weighted(gen);
		particles_sampled.push_back(particles[sample_index]);
		
	}
	
	particles = particles_sampled;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
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
