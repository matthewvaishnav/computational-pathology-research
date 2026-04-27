/**
 * History Screen
 * 
 * View past inference results
 */

import React, { useEffect, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  Image,
  useColorScheme,
} from 'react-native';
import { StackNavigationProp } from '@react-navigation/stack';
import { RootStackParamList } from '../../App';
import { getHistory, HistoryItem } from '../services/StorageService';

type HistoryScreenNavigationProp = StackNavigationProp<RootStackParamList, 'History'>;

interface Props {
  navigation: HistoryScreenNavigationProp;
}

const HistoryScreen: React.FC<Props> = ({ navigation }) => {
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const isDarkMode = useColorScheme() === 'dark';

  useEffect(() => {
    loadHistory();
  }, []);

  const loadHistory = async () => {
    const items = await getHistory();
    setHistory(items);
  };

  const renderItem = ({ item }: { item: HistoryItem }) => (
    <TouchableOpacity
      style={[styles.card, { backgroundColor: isDarkMode ? '#2a2a2a' : '#f8f8f8' }]}
      onPress={() => navigation.navigate('Results', {
        predictions: item.predictions,
        imageUri: item.imageUri,
      })}>
      <Image source={{ uri: item.imageUri }} style={styles.thumbnail} />
      <View style={styles.cardContent}>
        <Text style={[styles.cardTitle, { color: isDarkMode ? '#ffffff' : '#000000' }]}>
          {item.predictions[0]?.label || 'Unknown'}
        </Text>
        <Text style={[styles.cardSubtitle, { color: isDarkMode ? '#cccccc' : '#666666' }]}>
          {new Date(item.timestamp).toLocaleDateString()}
        </Text>
        <Text style={[styles.cardConfidence, { color: '#4CAF50' }]}>
          {(item.predictions[0]?.confidence * 100).toFixed(1)}% confidence
        </Text>
      </View>
    </TouchableOpacity>
  );

  return (
    <View style={[styles.container, { backgroundColor: isDarkMode ? '#1a1a1a' : '#ffffff' }]}>
      {history.length === 0 ? (
        <View style={styles.emptyContainer}>
          <Text style={[styles.emptyText, { color: isDarkMode ? '#cccccc' : '#666666' }]}>
            No history yet
          </Text>
          <Text style={[styles.emptySubtext, { color: isDarkMode ? '#aaaaaa' : '#888888' }]}>
            Analyze images to see them here
          </Text>
        </View>
      ) : (
        <FlatList
          data={history}
          renderItem={renderItem}
          keyExtractor={(item) => item.id}
          contentContainerStyle={styles.list}
        />
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  list: {
    padding: 15,
  },
  card: {
    flexDirection: 'row',
    padding: 15,
    borderRadius: 12,
    marginBottom: 15,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  thumbnail: {
    width: 80,
    height: 80,
    borderRadius: 8,
    marginRight: 15,
  },
  cardContent: {
    flex: 1,
    justifyContent: 'center',
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 5,
  },
  cardSubtitle: {
    fontSize: 14,
    marginBottom: 5,
  },
  cardConfidence: {
    fontSize: 14,
    fontWeight: '600',
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  emptyText: {
    fontSize: 20,
    fontWeight: '600',
    marginBottom: 10,
  },
  emptySubtext: {
    fontSize: 16,
  },
});

export default HistoryScreen;
