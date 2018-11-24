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

	num_particles = 150;

	// This lines creates a normal (Gaussian) distribution for x, y and theta
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	default_random_engine gen;

	for(int i=0; i<=num_particles; i++){
		Particle p;
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0;

		particles.push_back(p);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	for (int i=0; i<=num_particles; i++){
		float x_0 = particles[i].x;
		float y_0 = particles[i].y;
		float theta_0 = particles[i].theta;
		if (yaw_rate > 0.00001){
			particles[i].x = x_0 + (velocity / yaw_rate) * (sin(theta_0 + yaw_rate * delta_t) - sin(theta_0));
			particles[i].y = y_0 + (velocity / yaw_rate) * (cos(theta_0) - cos(theta_0 + yaw_rate * delta_t));
			particles[i].theta = theta_0 + yaw_rate * delta_t;
		}
		else{
			particles[i].x = x_0 + velocity * delta_t * cos(theta_0);
			particles[i].y = y_0 + velocity * delta_t * sin(theta_0);
		}
		// add noise with radom noise generators
		normal_distribution<double> dist_x(0, std_pos[0]);
		normal_distribution<double> dist_y(0, std_pos[1]);
		normal_distribution<double> dist_theta(0, std_pos[2]);
		default_random_engine gen;
		particles[i].x += dist_x(gen);
		particles[i].y += dist_y(gen);
		particles[i].theta += dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for (int i; i < observations.size(); i++){

		LandmarkObs obs = observations[i];
		double min_dist = 1e44; // init min_dist to a very large value

		int map_id = -1; // init the map id to -1

		for (int j; j<predicted.size(); j++){

			LandmarkObs	pred = predicted[j];
			double temp_dist = dist(obs.x, obs.y, pred.x, pred.y);
			if (temp_dist < min_dist){
				min_dist = temp_dist;
				map_id = pred.id;
			}
		}

		observations[i].id = map_id; // set to closest measurement map id to the observation
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

	for (int i=0; i<particles.size(); i++){ // for each particle
		Particle particle = particles[i];
		
		vector<LandmarkObs> predictions; // initialize a list of all the valid landmarks whithin sensor range
		for (int j=0; j<map_landmarks.landmark_list.size(); j++){
			double x_landmark = map_landmarks.landmark_list[j].x_f;
			double y_landmark = map_landmarks.landmark_list[j].y_f;
			double id_landmark = map_landmarks.landmark_list[j].id_i;

			if (dist(particle.x, particle.y, x_landmark, y_landmark) <= sensor_range){
				predictions.push_back(LandmarkObs{ id_landmark, x_landmark, y_landmark });
			}
		}

		// convert observation from vehicle coordinates to map coordinates
		vector<LandmarkObs> obs_tran; // (observations transformed)
		for(int j=0; j<observations.size(); j++){
			LandmarkObs obs_orig = observations[j];
			double x_obs = particle.x + cos(particle.theta) * obs_orig.x - sin(particle.theta) * obs_orig.y;
			double y_obs = particle.y + sin(particle.theta) * obs_orig.x + cos(particle.theta) * obs_orig.y;
			obs_tran.push_back(LandmarkObs{obs_orig.id, x_obs, y_obs});
		}
		// find the colsest landmark to each observation
		dataAssociation(predictions, obs_tran);

		// calculate the probability of each particle according to multivariant gaussian distribution
		double gauss_norm = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
		
		particles[i].weight = 1.0; // reinitialize the particle weight
		for (int j; j<obs_tran.size(); j++){
			LandmarkObs obs = obs_tran[j];
			// get the x,y coordinates of the prediction associated with the current observation
			for (int k = 0; k < predictions.size(); k++) {
				if (predictions[k].id == obs.id) {
					LandmarkObs land_mark = predictions[k];
				}
			}
			double exponent = pow((obs.x-land_mark.x)/std_landmark[0], 2) / 2;
			exponent += pow((obs.y-land_mark.y)/std_landmark[1], 2) / 2;
			// particle weight would bw the combination of the probabilities of all observations
			particles[i].weight *= gauss_norm * exp(-exponent);
		}

	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	vector<Particle> resampled_particles;
	double max_w = 0;
	for (int i=0; i<particles.size(); i++){
		// find the biggest weight
		if (particles[i].weight > max_w){
			max_w = particles[i].weight;
		}
	}
	uniform_real_distribution<double> dist_beta(0, max_w);
	uniform_int_distribution<int> dist_index(0, num_particles-1);
	default_random_engine gen;
	double beta;
	int idx = dist_index(gen); // generate random starting point
	for (int i=0; i<num_particles; i++){
		beta = 2 * dist_beta(gen);
		while (particles[idx].weight < beta){
			beta -= particles[idx].weight;
			idx = (idx+1) % num_particles;

		resampled_particles.push_back(particles[idx]);
		}
	}
	particles = resampled_particles;
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
