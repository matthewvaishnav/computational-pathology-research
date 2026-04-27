/**
 * Inference Screen
 * 
 * On-device AI inference processing
 */

import React, { useEffect, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ActivityIndicator,
  useColorScheme,
} from 'react-native';
import { StackNavigationProp } from '@react-navigation/stack';
import { RouteProp } from '@react-navigation/native';
import { RootStackParamList } from '../../App';
import { runInference } from '../services/InferenceService';

type InferenceScreenNavigationProp = StackNavigationProp<RootStackParamList, 'Inference'>;
type InferenceScreenRouteProp = RouteProp<RootStackParamList, 'Inference'>;

interface Props {
  navigation: InferenceScreenNavigationProp;
  route: InferenceScreenRouteProp;
}

const InferenceScreen: React.FC<Props> = ({ navigation, route }) => {
  const { imageUri } = route.params;
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState('Loading model...');
  const isDarkMode = useColorScheme() === 'dark';

  useEffect(() => {
    performInference();
  }, []);

  const performInference = async () => {
    try {
      // Load model
      setStatus('Loading model...');
      setProgress(0.2);
      await new Promise(resolve => setTimeout(resolve, 500));

      // Preprocess image
      setStatus('Preprocessing image...');
      setProgress(0.4);
      await new Promise(resolve => setTimeout(resolve, 300));

      // Run inference
      setStatus('Running AI inference...');
      setProgress(0.6);
      const predictions = await runInference(imageUri);

      // Postprocess
      setStatus('Analyzing results...');
      setProgress(0.8);
      await new Promise(resolve => setTimeout(resolve, 300));

      // Complete
      setProgress(1.0);
      navigation.replace('Results', { predictions, imageUri });

    } catch (error) {
      console.error('Inference error:', error);
      setStatus('Error: ' + (error as Error).message);
    }
  };

  return (
    <View style={[styles.container, { backgroundColor: isDarkMode ? '#1a1a1a' : '#ffffff' }]}>
      <View style={styles.content}>
        <ActivityIndicator size="large" color={isDarkMode ? '#ffffff' : '#007AFF'} />
        <Text style={[styles.status, { color: isDarkMode ? '#ffffff' : '#000000' }]}>
          {status}
        </Text>
        <View style={styles.progressContainer}>
          <View
            style={[
              styles.progressBar,
              {
                width: `${progress * 100}%`,
                backgroundColor: isDarkMode ? '#007AFF' : '#007AFF',
              },
            ]}
          />
        </View>
        <Text style={[styles.progressText, { color: isDarkMode ? '#cccccc' : '#666666' }]}>
          {Math.round(progress * 100)}%
        </Text>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  content: {
    alignItems: 'center',
    width: '100%',
  },
  status: {
    fontSize: 18,
    marginTop: 20,
    marginBottom: 30,
  },
  progressContainer: {
    width: '100%',
    height: 8,
    backgroundColor: '#e0e0e0',
    borderRadius: 4,
    overflow: 'hidden',
  },
  progressBar: {
    height: '100%',
    borderRadius: 4,
  },
  progressText: {
    fontSize: 14,
    marginTop: 10,
  },
});

export default InferenceScreen;
