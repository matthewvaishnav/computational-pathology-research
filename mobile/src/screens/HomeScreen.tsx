/**
 * Home Screen
 * 
 * Main landing screen with navigation options
 */

import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  useColorScheme,
} from 'react-native';
import { StackNavigationProp } from '@react-navigation/stack';
import { RootStackParamList } from '../../App';

type HomeScreenNavigationProp = StackNavigationProp<RootStackParamList, 'Home'>;

interface Props {
  navigation: HomeScreenNavigationProp;
}

const HomeScreen: React.FC<Props> = ({ navigation }) => {
  const isDarkMode = useColorScheme() === 'dark';

  const buttonStyle = [
    styles.button,
    { backgroundColor: isDarkMode ? '#3a3a3a' : '#007AFF' },
  ];

  const textStyle = [
    styles.buttonText,
    { color: '#ffffff' },
  ];

  return (
    <View style={[styles.container, { backgroundColor: isDarkMode ? '#1a1a1a' : '#ffffff' }]}>
      <View style={styles.header}>
        <Text style={[styles.title, { color: isDarkMode ? '#ffffff' : '#000000' }]}>
          Medical AI Pathology
        </Text>
        <Text style={[styles.subtitle, { color: isDarkMode ? '#cccccc' : '#666666' }]}>
          AI-powered pathology analysis on your device
        </Text>
      </View>

      <View style={styles.buttonContainer}>
        <TouchableOpacity
          style={buttonStyle}
          onPress={() => navigation.navigate('Camera')}>
          <Text style={textStyle}>📷 Capture Image</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={buttonStyle}
          onPress={() => navigation.navigate('History')}>
          <Text style={textStyle}>📋 View History</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={buttonStyle}
          onPress={() => navigation.navigate('Settings')}>
          <Text style={textStyle}>⚙️ Settings</Text>
        </TouchableOpacity>
      </View>

      <View style={styles.footer}>
        <Text style={[styles.footerText, { color: isDarkMode ? '#888888' : '#999999' }]}>
          Offline-first • Privacy-focused • Medical-grade
        </Text>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
  },
  header: {
    marginTop: 40,
    marginBottom: 60,
    alignItems: 'center',
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  subtitle: {
    fontSize: 16,
    textAlign: 'center',
  },
  buttonContainer: {
    flex: 1,
    justifyContent: 'center',
    gap: 20,
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
  },
  footer: {
    marginTop: 40,
    alignItems: 'center',
  },
  footerText: {
    fontSize: 12,
  },
});

export default HomeScreen;
