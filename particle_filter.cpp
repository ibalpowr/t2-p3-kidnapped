/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 *  Modified on: Nov 8, 2017
 *      Author: Joe Zhou
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

  //100 is a good start, but adjust it as see fit
  num_particles = 100;
  
  //update size if needed
  particles.resize(num_particles);
  weights.resize(num_particles);

  // for random normal noise generation
  default_random_engine gen;
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  // sum of all weights is 1
  double initial_weight = 1.0/num_particles;

  for (int i = 0; i < num_particles; ++i) {
    // Particle is a struct
    Particle &p = particles[i];
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = initial_weight;

    weights[i] = initial_weight;
  }

  //when done, change flag
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  //random normal noise
  default_random_engine gen;
  normal_distribution<double> dist_x(0.0, std_pos[0]);
  normal_distribution<double> dist_y(0.0, std_pos[1]);
  normal_distribution<double> dist_theta(0.0, std_pos[2]);
    
  //add measurements to each particles
  for(int i=0; i<num_particles; i++){
    Particle &p = particles[i];

    //if yaw rate is or near zero
    if(fabs(yaw_rate)<0.0001){
      p.x += velocity * delta_t * cos(p.theta);
      p.y += velocity * delta_t * sin(p.theta);
    }else{
      p.x += (velocity/yaw_rate) * (sin(p.theta + yaw_rate*delta_t) -
             sin(p.theta));
      p.y += (velocity/yaw_rate) * (cos(p.theta) - cos(p.theta + 
             yaw_rate * delta_t));
      p.theta += yaw_rate * delta_t;
    }
        
    //add noise
    p.x += dist_x(gen);
    p.y += dist_y(gen);
    p.theta += dist_theta(gen);
  } 
}

//this method is not called
//find nearest neighbor
void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, 
                                     std::vector<LandmarkObs>& observations) {
  for (LandmarkObs& observation: observations) {
    double dist_to_predict = -1; 
    int predict_id = -1;
    for (LandmarkObs predict : predicted) {
      double distance = dist(observation.x, observation.y, 
                        predict.x, predict.y);
      if (dist_to_predict == -1 || distance < dist_to_predict) {
        dist_to_predict = distance;
        predict_id = predict.id;
      }
    }
    observation.id = predict_id;
  }
}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                        const std::vector<LandmarkObs> &observations, 
                        const Map &map_landmarks) {
                        
  double sum_weights = 0.0;

  for(int i = 0; i < num_particles; i++){

    double weight = 1.0;

    double multiplier = 1 / (2.0 * M_PI * std_landmark[0] * std_landmark[1]);
    double pow1 = pow(std_landmark[0], 2);
    double pow2 = pow(std_landmark[1], 2);

    for (int j = 0; j < observations.size(); j++){

      //1st: transform observed landmarks from the car's coordinates to 
      //the map's coordinates,in respect to particle[i]

      LandmarkObs obs_landmark;

      obs_landmark.id = observations[j].id;
      obs_landmark.x = particles[i].x + 
                        observations[j].x * cos(particles[i].theta) - 
                        observations[j].y * sin(particles[i].theta);
      obs_landmark.y = particles[i].y + 
                        observations[j].x * sin(particles[i].theta) +
                        observations[j].y * cos(particles[i].theta);

      //2nd: associate the transformed landmarks with map landmarks

      default_random_engine gen;
      bernoulli_distribution distribution(0.5);

      Map::single_landmark_s closest_landmark = map_landmarks.landmark_list[0];

      double min_distance = dist(obs_landmark.x, obs_landmark.y, 
                              closest_landmark.x_f, closest_landmark.y_f);

      for(size_t j = 1; j<map_landmarks.landmark_list.size(); j++){

        Map::single_landmark_s current_landmark=map_landmarks.landmark_list[j];

        double cur_distance = dist(obs_landmark.x, obs_landmark.y, 
                               current_landmark.x_f, current_landmark.y_f);

        // add sensor_range restriction
        if(cur_distance <= sensor_range & cur_distance < min_distance ||
                        (cur_distance == min_distance && distribution(gen))){

          closest_landmark = map_landmarks.landmark_list[j];
          min_distance = cur_distance;
        }
      }

      //3rd: calculate the weight of a single observed landmark in respect to 
      //the closest landmark if particle[i] is where the car is

      weight *= multiplier *
          exp(-0.5 * (pow((closest_landmark.x_f - obs_landmark.x), 2) / pow1 +
          pow((closest_landmark.y_f - obs_landmark.y), 2) / pow2));
    }

    //4th: get the final weight
      sum_weights += weight;
      particles[i].weight = weight;
  }

  //update weights and particles.
  for(int i = 0; i < num_particles; i++){
    particles[i].weight /= sum_weights;
    weights[i] = particles[i].weight;
  } 
}

//resample particles w/ replacement w/ prob proportional to their weights
void ParticleFilter::resample() {

  std::vector<Particle> particles_old(particles);
  default_random_engine gen;

  discrete_distribution<std::size_t> d(weights.begin(), weights.end());

  for(Particle& p: particles){
    p = particles_old[d(gen)];
  }
}

Particle ParticleFilter::SetAssociations(Particle particle, 
        std::vector<int> associations, std::vector<double> sense_x, 
        std::vector<double> sense_y) {
  //particle: the particle to assign each listed association, and 
  //association's (x,y) world coordinates mapping to
  //associations: The landmark id that goes along with each listed association
  //sense_x: the associations x mapping already converted to world coordinates
  //sense_y: the associations y mapping already converted to world coordinates

  //Clear the previous associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

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
