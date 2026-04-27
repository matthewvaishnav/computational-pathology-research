/**
 * Results Screen
 * 
 * Display AI inference results
 */

import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Image,
  TouchableOpacity,
  useColorScheme,
} from 'react-native';
import { StackNavigationProp } from '@react-navigation/stack';
import { RouteProp } from '@react-navigation/native';
import { RootStackParamList } from '../../App';

type ResultsScreenNavigationProp = StackNavigationProp<RootStackParamList, 'Results'>;
type ResultsScreenRouteProp = RouteProp<RootStackParamList, 'Results'>;

interface Props {
  navigation: ResultsScreenNavigationProp;
  route: ResultsScreenRouteProp;
}

const ResultsScreen: React.FC<Props> = ({ navigation, route }) => {
  const { predictions, imageUri } = route.params;
  const isDarkMode = useColorScheme() === 'dark';

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return '#4CAF50';
    if (confidence >= 0.6) return '#FF9800';
    return '#F44336';
  };

  return (
    <ScrollView
      style={[styles.container, { backgroundColor: isDarkMode ? '#1a1a1a' : '#ffffff' }]}>
      <Image source={{ uri: imageUri }} style={styles.image} />

      <View style={styles.resultsContainer}>
        <Text style={[styles.title, { color: isDarkMode ? '#ffffff' : '#000000' }]}>
          Analysis Results
        </Text>

        {predictions.map((pred: any, index: number) => (
          <View
            key={index}
            style={[
              styles.predictionCard,
              { backgroundColor: isDarkMode ? '#2a2a2a' : '#f8f8f8' },
            ]}>
            <View style={styles.predictionHeader}>
              <Text style={[styles.label, { color: isDarkMode ? '#ffffff' : '#000000' }]}>
                {pred.label}
              </Text>
              <Text
                style={[
                  styles.confidence,
                  { color: getConfidenceColor(pred.confidence) },
                ]}>
                {(pred.confidence * 100).toFixed(1)}%
              </Text>
            </View>
            <View
              style={[
                styles.confidenceBar,
                { backgroundColor: isDarkMode ? '#3a3a3a' : '#e0e0e0' },
              ]}>
              <View
                style={[
                  styles.confidenceFill,
                  {
                    width: `${pred.confidence * 100}%`,
                    backgroundColor: getConfidenceColor(pred.confidence),
                  },
                ]}
              />
            </View>
          </View>
        ))}

        <View style={styles.metadataContainer}>
          <Text style={[styles.metadataTitle, { color: isDarkMode ? '#cccccc' : '#666666' }]}>
            Inference Details
          </Text>
          <Text style={[styles.metadata, { color: isDarkMode ? '#aaaaaa' : '#888888' }]}>
            Model: Medical AI Foundation v1.0
          </Text>
          <Text style={[styles.metadata, { color: isDarkMode ? '#aaaaaa' : '#888888' }]}>
            Inference Time: {predictions.inference_time_ms?.toFixed(0) || 'N/A'} ms
          </Text>
          <Text style={[styles.metadata, { color: isDarkMode ? '#aaaaaa' : '#888888' }]}>
            Device: On-device processing
          </Text>
        </View>

        <View style={styles.disclaimer}>
          <Text style={[styles.disclaimerText, { color: isDarkMode ? '#888888' : '#999999' }]}>
            ⚠️ For research purposes only. Not for clinical diagnosis.
          </Text>
        </View>

        <TouchableOpacity
          style={[styles.button, { backgroundColor: isDarkMode ? '#3a3a3a' : '#007AFF' }]}
          onPress={() => navigation.navigate('Home')}>
          <Text style={styles.buttonText}>Done</Text>
        </TouchableOpacity>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  image: {
    width: '100%',
    height: 300,
  },
  resultsContainer: {
    padding: 20,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 20,
  },
  predictionCard: {
    padding: 15,
    borderRadius: 12,
    marginBottom: 15,
  },
  predictionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  label: {
    fontSize: 18,
    fontWeight: '600',
  },
  confidence: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  confidenceBar: {
    height: 8,
    borderRadius: 4,
    overflow: 'hidden',
  },
  confidenceFill: {
    height: '100%',
  },
  metadataContainer: {
    marginTop: 20,
    marginBottom: 20,
  },
  metadataTitle: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 10,
  },
  metadata: {
    fontSize: 14,
    marginBottom: 5,
  },
  disclaimer: {
    padding: 15,
    backgroundColor: '#FFF3CD',
    borderRadius: 8,
    marginBottom: 20,
  },
  disclaimerText: {
    fontSize: 12,
    textAlign: 'center',
  },
  button: {
    padding: 20,
    borderRadius: 12,
    alignItems: 'center',
  },
  buttonText: {
    fontSize: 18,
    fontWeight: '600',
    color: '#ffffff',
  },
});

export default ResultsScreen;
