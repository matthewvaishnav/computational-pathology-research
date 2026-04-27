/**
 * Settings Screen
 * 
 * App configuration and preferences
 */

import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Switch,
  TouchableOpacity,
  useColorScheme,
} from 'react-native';
import { getSettings, saveSettings, Settings } from '../services/StorageService';

const SettingsScreen: React.FC = () => {
  const [settings, setSettings] = useState<Settings>({
    offlineMode: true,
    saveHistory: true,
    highQualityInference: false,
    autoSync: false,
  });
  const isDarkMode = useColorScheme() === 'dark';

  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    const saved = await getSettings();
    setSettings(saved);
  };

  const updateSetting = async (key: keyof Settings, value: boolean) => {
    const updated = { ...settings, [key]: value };
    setSettings(updated);
    await saveSettings(updated);
  };

  const SettingRow = ({
    title,
    subtitle,
    value,
    onValueChange,
  }: {
    title: string;
    subtitle: string;
    value: boolean;
    onValueChange: (value: boolean) => void;
  }) => (
    <View style={[styles.settingRow, { backgroundColor: isDarkMode ? '#2a2a2a' : '#f8f8f8' }]}>
      <View style={styles.settingText}>
        <Text style={[styles.settingTitle, { color: isDarkMode ? '#ffffff' : '#000000' }]}>
          {title}
        </Text>
        <Text style={[styles.settingSubtitle, { color: isDarkMode ? '#cccccc' : '#666666' }]}>
          {subtitle}
        </Text>
      </View>
      <Switch
        value={value}
        onValueChange={onValueChange}
        trackColor={{ false: '#767577', true: '#81b0ff' }}
        thumbColor={value ? '#007AFF' : '#f4f3f4'}
      />
    </View>
  );

  return (
    <ScrollView
      style={[styles.container, { backgroundColor: isDarkMode ? '#1a1a1a' : '#ffffff' }]}>
      <View style={styles.section}>
        <Text style={[styles.sectionTitle, { color: isDarkMode ? '#cccccc' : '#666666' }]}>
          Inference
        </Text>
        <SettingRow
          title="Offline Mode"
          subtitle="Run all inference on-device"
          value={settings.offlineMode}
          onValueChange={(value) => updateSetting('offlineMode', value)}
        />
        <SettingRow
          title="High Quality Inference"
          subtitle="Use full precision model (slower)"
          value={settings.highQualityInference}
          onValueChange={(value) => updateSetting('highQualityInference', value)}
        />
      </View>

      <View style={styles.section}>
        <Text style={[styles.sectionTitle, { color: isDarkMode ? '#cccccc' : '#666666' }]}>
          Data
        </Text>
        <SettingRow
          title="Save History"
          subtitle="Store inference results locally"
          value={settings.saveHistory}
          onValueChange={(value) => updateSetting('saveHistory', value)}
        />
        <SettingRow
          title="Auto Sync"
          subtitle="Sync with cloud when available"
          value={settings.autoSync}
          onValueChange={(value) => updateSetting('autoSync', value)}
        />
      </View>

      <View style={styles.section}>
        <Text style={[styles.sectionTitle, { color: isDarkMode ? '#cccccc' : '#666666' }]}>
          About
        </Text>
        <View style={[styles.infoCard, { backgroundColor: isDarkMode ? '#2a2a2a' : '#f8f8f8' }]}>
          <Text style={[styles.infoText, { color: isDarkMode ? '#ffffff' : '#000000' }]}>
            Medical AI Pathology v1.0.0
          </Text>
          <Text style={[styles.infoSubtext, { color: isDarkMode ? '#cccccc' : '#666666' }]}>
            AI-powered pathology analysis
          </Text>
          <Text style={[styles.infoSubtext, { color: isDarkMode ? '#cccccc' : '#666666' }]}>
            For research purposes only
          </Text>
        </View>
      </View>

      <TouchableOpacity
        style={[styles.button, { backgroundColor: isDarkMode ? '#3a3a3a' : '#007AFF' }]}>
        <Text style={styles.buttonText}>Clear All Data</Text>
      </TouchableOpacity>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  section: {
    marginTop: 20,
    paddingHorizontal: 15,
  },
  sectionTitle: {
    fontSize: 14,
    fontWeight: '600',
    marginBottom: 10,
    textTransform: 'uppercase',
  },
  settingRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 15,
    borderRadius: 12,
    marginBottom: 10,
  },
  settingText: {
    flex: 1,
    marginRight: 15,
  },
  settingTitle: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 5,
  },
  settingSubtitle: {
    fontSize: 14,
  },
  infoCard: {
    padding: 15,
    borderRadius: 12,
  },
  infoText: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 5,
  },
  infoSubtext: {
    fontSize: 14,
    marginBottom: 3,
  },
  button: {
    margin: 15,
    padding: 15,
    borderRadius: 12,
    alignItems: 'center',
  },
  buttonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#ffffff',
  },
});

export default SettingsScreen;
