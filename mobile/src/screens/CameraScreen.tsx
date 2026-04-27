/**
 * Camera Screen
 * 
 * Image capture and selection
 */

import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Image,
  Alert,
  useColorScheme,
} from 'react-native';
import { StackNavigationProp } from '@react-navigation/stack';
import { launchCamera, launchImageLibrary, ImagePickerResponse } from 'react-native-image-picker';
import { RootStackParamList } from '../../App';

type CameraScreenNavigationProp = StackNavigationProp<RootStackParamList, 'Camera'>;

interface Props {
  navigation: CameraScreenNavigationProp;
}

const CameraScreen: React.FC<Props> = ({ navigation }) => {
  const [imageUri, setImageUri] = useState<string | null>(null);
  const isDarkMode = useColorScheme() === 'dark';

  const handleCameraLaunch = () => {
    launchCamera(
      {
        mediaType: 'photo',
        quality: 1,
        saveToPhotos: true,
      },
      (response: ImagePickerResponse) => {
        if (response.didCancel) {
          return;
        }
        if (response.errorCode) {
          Alert.alert('Error', response.errorMessage || 'Camera error');
          return;
        }
        if (response.assets && response.assets[0].uri) {
          setImageUri(response.assets[0].uri);
        }
      }
    );
  };

  const handleGalleryLaunch = () => {
    launchImageLibrary(
      {
        mediaType: 'photo',
        quality: 1,
      },
      (response: ImagePickerResponse) => {
        if (response.didCancel) {
          return;
        }
        if (response.errorCode) {
          Alert.alert('Error', response.errorMessage || 'Gallery error');
          return;
        }
        if (response.assets && response.assets[0].uri) {
          setImageUri(response.assets[0].uri);
        }
      }
    );
  };

  const handleAnalyze = () => {
    if (imageUri) {
      navigation.navigate('Inference', { imageUri });
    }
  };

  const buttonStyle = [
    styles.button,
    { backgroundColor: isDarkMode ? '#3a3a3a' : '#007AFF' },
  ];

  return (
    <View style={[styles.container, { backgroundColor: isDarkMode ? '#1a1a1a' : '#ffffff' }]}>
      {imageUri ? (
        <View style={styles.previewContainer}>
          <Image source={{ uri: imageUri }} style={styles.preview} />
          <View style={styles.actionButtons}>
            <TouchableOpacity
              style={[buttonStyle, styles.actionButton]}
              onPress={() => setImageUri(null)}>
              <Text style={styles.buttonText}>🔄 Retake</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[buttonStyle, styles.actionButton]}
              onPress={handleAnalyze}>
              <Text style={styles.buttonText}>✓ Analyze</Text>
            </TouchableOpacity>
          </View>
        </View>
      ) : (
        <View style={styles.captureContainer}>
          <Text style={[styles.instruction, { color: isDarkMode ? '#cccccc' : '#666666' }]}>
            Capture or select a pathology image
          </Text>
          <TouchableOpacity style={buttonStyle} onPress={handleCameraLaunch}>
            <Text style={styles.buttonText}>📷 Take Photo</Text>
          </TouchableOpacity>
          <TouchableOpacity style={buttonStyle} onPress={handleGalleryLaunch}>
            <Text style={styles.buttonText}>🖼️ Choose from Gallery</Text>
          </TouchableOpacity>
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
  },
  captureContainer: {
    flex: 1,
    justifyContent: 'center',
    gap: 20,
  },
  previewContainer: {
    flex: 1,
  },
  preview: {
    flex: 1,
    borderRadius: 12,
    marginBottom: 20,
  },
  instruction: {
    fontSize: 16,
    textAlign: 'center',
    marginBottom: 20,
  },
  button: {
    padding: 20,
    borderRadius: 12,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  buttonText: {
    fontSize: 18,
    fontWeight: '600',
    color: '#ffffff',
  },
  actionButtons: {
    flexDirection: 'row',
    gap: 10,
  },
  actionButton: {
    flex: 1,
  },
});

export default CameraScreen;
