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
default_random_engine gen;

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

	for(int i=0; i < num_particles; i++){
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

	double x, y, t;
	for (int i=0; i<num_particles; i++){
		x = particles[i].x;
		y = particles[i].y;
		t = particles[i].theta;
		if (fabs(yaw_rate) > 0.001){
			x += velocity / yaw_rate * (sin(t + yaw_rate * delta_t) - sin(t));
			y += velocity / yaw_rate * (cos(t) - cos(t + yaw_rate * delta_t));
		}
		else{
			x += velocity * delta_t * cos(t + yaw_rate * delta_t / 2);
			y += velocity * delta_t * sin(t + yaw_rate * delta_t / 2);
		}
		t += yaw_rate * delta_t;
		
		// add noise with radom noise generators and update the particles
		particles[i].x = normal_distribution<double>(x, std_pos[0])(gen);
		particles[i].y = normal_distribution<double>(y, std_pos[1])(gen);
		particles[i].theta = normal_distribution<double>(t, std_pos[2])(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for (unsigned int i; i < observations.size(); i++){

		LandmarkObs obs = observations[i];
		double min_dist = 1e99; // init min_dist to a very large value

		int map_id = -1; // init the map id to -1

		for (unsigned int j=0; j<predicted.size(); j++){

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

	double w_norm = 0; // will be used to normalize the weights in 0-1 range
	
	for (int i=0; i<num_particles; i++){ // for each particle

		particles[i].weight = 1.0; // reinitialize the particle weight

		double x_p = particles[i].x;
		double y_p = particles[i].y;
		double t_p = particles[i].theta;

		// convert observation from vehicle coordinates to map coordinates
		vector<LandmarkObs> obs_tran; // (observations transformed)
		for(unsigned int j=0; j<observations.size(); j++){
			double x_tran = x_p + cos(t_p) * observations[j].x - sin(t_p) * observations[j].y;
			double y_tran = y_p + sin(t_p) * observations[j].x + cos(t_p) * observations[j].y;
			int id_tran = observations[j].id;
			obs_tran.push_back(LandmarkObs{id_tran, x_tran, y_tran});
		}

		// initialize a list of all the valid landmarks whithin sensor range
		vector<LandmarkObs> valid_preds; 
		for (unsigned int j=0; j<map_landmarks.landmark_list.size(); j++){
			double x_landmark = map_landmarks.landmark_list[j].x_f;
			double y_landmark = map_landmarks.landmark_list[j].y_f;
			int id_landmark = map_landmarks.landmark_list[j].id_i;
			
			// take only landmarks within sensor range
			double R = sqrt(pow(x_landmark-x_p, 2) + pow(y_landmark-y_p, 2));
			if (R < sensor_range){
				valid_preds.push_back(LandmarkObs{ id_landmark, x_landmark, y_landmark });
			}
		}

		if (valid_preds.size() == 0){
			particles[i].weight = 0;
			cout << "no valid predictions\n";		
			continue;
		}

		// find the colsest landmark to each observation
		dataAssociation(valid_preds, obs_tran);

		// calculate the probability of each particle according to multivariant gaussian distribution
		double gauss_norm = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
		
		vector<int> associations;
		vector<double> sense_x, sense_y;

		for (unsigned int j=0; j<obs_tran.size(); j++){
			double x_obs, y_obs, x_pred, y_pred;
			x_obs = obs_tran[j].x;
			y_obs = obs_tran[j].y;
			int target_id = obs_tran[j].id;
			//cout <<"target id: "<<target_id<<"\n";
			// get the x,y coordinates of the prediction associated with the current observation
			for (unsigned int k = 0; k < valid_preds.size(); k++) {
				//cout <<"prediction id: "<<valid_preds[k].id<<"\n";
				if (valid_preds[k].id == target_id) {
					//cout << "found a match!\n";
					x_pred = valid_preds[k].x;
					y_pred = valid_preds[k].y;
					associations.push_back(valid_preds[k].id);
					sense_x.push_back(x_obs);
					sense_y.push_back(y_obs);
					break;
				}
			}
			double dx = x_obs - x_pred;
			double dy = y_obs - y_pred;
			double exponent = -0.5 * (pow(dx / std_landmark[0], 2) + pow(dy / std_landmark[1], 2));

			// particle weight would bw the combination of the probabilities of all observations
			particles[i].weight *= gauss_norm * exp(exponent);
		}

		w_norm += particles[i].weight;
		SetAssociations(particles[i], associations, sense_x, sense_y);
	}
	// normalize all weights between 0-1
	for (unsigned int i=0; i < particles.size(); i++){
		if (w_norm == 0) {
			particles[i].weight = 1. / num_particles;
		}
		else {
			particles[i].weight /= w_norm;
		}
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	vector<Particle> resampled_particles;
	double max_w = 0;
	for (unsigned int i=0; i<particles.size(); i++){
		// find the biggest weight
		if (particles[i].weight > max_w){
			max_w = particles[i].weight;
		}
	}
	uniform_real_distribution<double> dist_beta(0, max_w);
	uniform_int_distribution<int> dist_index(0, num_particles-1);
	double beta;
	int idx = dist_index(gen); // generate random starting point
	for (int i=0; i<num_particles; i++){
		beta = 2 * dist_beta(gen);
		while (particles[idx].weight < beta){
			beta -= particles[idx].weight;
			idx = (idx+1) % num_particles;
		}
		resampled_particles.push_back(particles[idx]);
	}
	particles = resampled_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
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
