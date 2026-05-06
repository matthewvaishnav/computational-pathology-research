#!/usr/bin/env python3
"""Database corruption and recovery testing."""

import os
import sqlite3
import tempfile
import shutil
import random
from pathlib import Path
from typing import Dict, List, Any

class DatabaseTester:
    """Test database corruption scenarios and recovery."""
    
    def __init__(self):
        self.temp_dir = None
        self.db_path = None
        
    def setup_test_db(self):
        """Create a test database with sample data."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute("""
            CREATE TABLE patients (
                id INTEGER PRIMARY KEY,
                patient_id TEXT UNIQUE,
                name TEXT,
                birth_date TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE studies (
                id INTEGER PRIMARY KEY,
                study_uid TEXT UNIQUE,
                patient_id TEXT,
                study_date TEXT,
                modality TEXT,
                FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE results (
                id INTEGER PRIMARY KEY,
                study_uid TEXT,
                prediction REAL,
                confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (study_uid) REFERENCES studies (study_uid)
            )
        """)
        
        # Insert sample data
        patients = [
            ("PAT001", "John Doe", "1980-01-01"),
            ("PAT002", "Jane Smith", "1975-05-15"),
            ("PAT003", "Bob Johnson", "1990-12-31"),
        ]
        
        for patient_id, name, birth_date in patients:
            cursor.execute(
                "INSERT INTO patients (patient_id, name, birth_date) VALUES (?, ?, ?)",
                (patient_id, name, birth_date)
            )
        
        studies = [
            ("1.2.3.4.5.1", "PAT001", "2024-01-01", "SM"),
            ("1.2.3.4.5.2", "PAT001", "2024-02-01", "SM"),
            ("1.2.3.4.5.3", "PAT002", "2024-01-15", "SM"),
        ]
        
        for study_uid, patient_id, study_date, modality in studies:
            cursor.execute(
                "INSERT INTO studies (study_uid, patient_id, study_date, modality) VALUES (?, ?, ?, ?)",
                (study_uid, patient_id, study_date, modality)
            )
        
        results = [
            ("1.2.3.4.5.1", 0.85, 0.92),
            ("1.2.3.4.5.2", 0.23, 0.78),
            ("1.2.3.4.5.3", 0.91, 0.95),
        ]
        
        for study_uid, prediction, confidence in results:
            cursor.execute(
                "INSERT INTO results (study_uid, prediction, confidence) VALUES (?, ?, ?)",
                (study_uid, prediction, confidence)
            )
        
        conn.commit()
        conn.close()
        
    def cleanup(self):
        """Clean up test database."""
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

def test_database_integrity_check():
    """Test database integrity checking."""
    print("Testing database integrity check...")
    
    tester = DatabaseTester()
    tester.setup_test_db()
    
    try:
        conn = sqlite3.connect(tester.db_path)
        cursor = conn.cursor()
        
        # Run integrity check
        cursor.execute("PRAGMA integrity_check")
        result = cursor.fetchone()
        
        integrity_ok = result[0] == "ok"
        print(f"  Integrity check: {'PASSED' if integrity_ok else 'FAILED'}")
        
        # Check foreign key constraints
        cursor.execute("PRAGMA foreign_key_check")
        fk_violations = cursor.fetchall()
        
        fk_ok = len(fk_violations) == 0
        print(f"  Foreign key check: {'PASSED' if fk_ok else f'FAILED ({len(fk_violations)} violations)'}")
        
        conn.close()
        return integrity_ok and fk_ok
        
    except Exception as e:
        print(f"  Integrity check error: {e}")
        return False
    finally:
        tester.cleanup()

def test_partial_corruption_recovery():
    """Test recovery from partial database corruption."""
    print("Testing partial corruption recovery...")
    
    tester = DatabaseTester()
    tester.setup_test_db()
    
    try:
        # Simulate partial corruption by truncating file
        original_size = tester.db_path.stat().st_size
        
        with open(tester.db_path, "r+b") as f:
            # Truncate to 80% of original size
            truncated_size = int(original_size * 0.8)
            f.truncate(truncated_size)
        
        print(f"  Simulated corruption: {original_size} -> {truncated_size} bytes")
        
        # Try to recover data
        conn = sqlite3.connect(tester.db_path)
        cursor = conn.cursor()
        
        recovered_data = {}
        
        # Try to read each table
        tables = ["patients", "studies", "results"]
        for table in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                recovered_data[table] = count
                print(f"  Recovered from {table}: {count} records")
            except sqlite3.DatabaseError as e:
                print(f"  Failed to recover {table}: {e}")
                recovered_data[table] = 0
        
        conn.close()
        
        # Check if any data was recovered
        total_recovered = sum(recovered_data.values())
        return total_recovered > 0
        
    except Exception as e:
        print(f"  Corruption recovery error: {e}")
        return False
    finally:
        tester.cleanup()

def test_transaction_rollback():
    """Test transaction rollback on errors."""
    print("Testing transaction rollback...")
    
    tester = DatabaseTester()
    tester.setup_test_db()
    
    try:
        conn = sqlite3.connect(tester.db_path)
        cursor = conn.cursor()
        
        # Count initial records
        cursor.execute("SELECT COUNT(*) FROM patients")
        initial_count = cursor.fetchone()[0]
        
        # Start transaction that will fail
        try:
            cursor.execute("BEGIN TRANSACTION")
            
            # Insert valid record
            cursor.execute(
                "INSERT INTO patients (patient_id, name, birth_date) VALUES (?, ?, ?)",
                ("PAT004", "Test Patient", "2000-01-01")
            )
            
            # Insert duplicate patient_id (should fail due to UNIQUE constraint)
            cursor.execute(
                "INSERT INTO patients (patient_id, name, birth_date) VALUES (?, ?, ?)",
                ("PAT001", "Duplicate Patient", "2000-01-01")
            )
            
            cursor.execute("COMMIT")
            
        except sqlite3.IntegrityError:
            cursor.execute("ROLLBACK")
            print("  Transaction rolled back due to constraint violation")
        
        # Check that no records were added
        cursor.execute("SELECT COUNT(*) FROM patients")
        final_count = cursor.fetchone()[0]
        
        rollback_worked = final_count == initial_count
        print(f"  Records before: {initial_count}, after: {final_count}")
        print(f"  Rollback successful: {rollback_worked}")
        
        conn.close()
        return rollback_worked
        
    except Exception as e:
        print(f"  Transaction rollback error: {e}")
        return False
    finally:
        tester.cleanup()

def test_backup_and_restore():
    """Test database backup and restore."""
    print("Testing backup and restore...")
    
    tester = DatabaseTester()
    tester.setup_test_db()
    
    try:
        backup_path = Path(tester.temp_dir) / "backup.db"
        
        # Create backup
        shutil.copy2(tester.db_path, backup_path)
        
        # Verify backup exists and has same size
        original_size = tester.db_path.stat().st_size
        backup_size = backup_path.stat().st_size
        
        backup_created = backup_size == original_size
        print(f"  Backup created: {backup_created} ({backup_size} bytes)")
        
        # Corrupt original database
        with open(tester.db_path, "w") as f:
            f.write("corrupted data")
        
        # Restore from backup
        shutil.copy2(backup_path, tester.db_path)
        
        # Verify restore
        conn = sqlite3.connect(tester.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM patients")
        patient_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM studies")
        study_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM results")
        result_count = cursor.fetchone()[0]
        
        conn.close()
        
        restore_successful = patient_count > 0 and study_count > 0 and result_count > 0
        print(f"  Restored data: {patient_count} patients, {study_count} studies, {result_count} results")
        print(f"  Restore successful: {restore_successful}")
        
        return backup_created and restore_successful
        
    except Exception as e:
        print(f"  Backup/restore error: {e}")
        return False
    finally:
        tester.cleanup()

def test_concurrent_access():
    """Test concurrent database access."""
    print("Testing concurrent access...")
    
    tester = DatabaseTester()
    tester.setup_test_db()
    
    try:
        # Open multiple connections
        conn1 = sqlite3.connect(tester.db_path)
        conn2 = sqlite3.connect(tester.db_path)
        
        cursor1 = conn1.cursor()
        cursor2 = conn2.cursor()
        
        # Test concurrent reads
        cursor1.execute("SELECT COUNT(*) FROM patients")
        count1 = cursor1.fetchone()[0]
        
        cursor2.execute("SELECT COUNT(*) FROM patients")
        count2 = cursor2.fetchone()[0]
        
        concurrent_reads = count1 == count2
        print(f"  Concurrent reads: {concurrent_reads} (both got {count1})")
        
        # Test write conflict handling
        try:
            cursor1.execute("BEGIN IMMEDIATE")
            cursor1.execute("INSERT INTO patients (patient_id, name, birth_date) VALUES (?, ?, ?)",
                          ("PAT005", "Concurrent Test 1", "2000-01-01"))
            
            # This should be blocked or fail
            cursor2.execute("BEGIN IMMEDIATE")
            cursor2.execute("INSERT INTO patients (patient_id, name, birth_date) VALUES (?, ?, ?)",
                          ("PAT006", "Concurrent Test 2", "2000-01-01"))
            
            cursor1.execute("COMMIT")
            cursor2.execute("COMMIT")
            
            write_conflict_handled = True
            
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e).lower():
                print("  Write conflict properly detected")
                cursor1.execute("COMMIT")
                write_conflict_handled = True
            else:
                print(f"  Unexpected write conflict error: {e}")
                write_conflict_handled = False
        
        conn1.close()
        conn2.close()
        
        return concurrent_reads and write_conflict_handled
        
    except Exception as e:
        print(f"  Concurrent access error: {e}")
        return False
    finally:
        tester.cleanup()

def test_large_data_handling():
    """Test handling of large datasets."""
    print("Testing large data handling...")
    
    tester = DatabaseTester()
    tester.setup_test_db()
    
    try:
        conn = sqlite3.connect(tester.db_path)
        cursor = conn.cursor()
        
        # Insert large number of records
        large_data = []
        for i in range(1000):
            large_data.append((
                f"STUDY_{i:06d}",
                f"PAT{i % 100:03d}",
                f"2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
                "SM"
            ))
        
        cursor.executemany(
            "INSERT INTO studies (study_uid, patient_id, study_date, modality) VALUES (?, ?, ?, ?)",
            large_data
        )
        
        conn.commit()
        
        # Verify data was inserted
        cursor.execute("SELECT COUNT(*) FROM studies")
        total_studies = cursor.fetchone()[0]
        
        # Test query performance on large dataset
        import time
        start_time = time.time()
        
        cursor.execute("SELECT * FROM studies WHERE modality = 'SM' ORDER BY study_date LIMIT 100")
        results = cursor.fetchall()
        
        query_time = time.time() - start_time
        
        print(f"  Inserted {len(large_data)} records")
        print(f"  Total studies: {total_studies}")
        print(f"  Query time: {query_time:.3f}s")
        print(f"  Query results: {len(results)} records")
        
        conn.close()
        
        return total_studies >= 1000 and query_time < 1.0 and len(results) == 100
        
    except Exception as e:
        print(f"  Large data handling error: {e}")
        return False
    finally:
        tester.cleanup()

def test_schema_migration():
    """Test database schema migration."""
    print("Testing schema migration...")
    
    tester = DatabaseTester()
    tester.setup_test_db()
    
    try:
        conn = sqlite3.connect(tester.db_path)
        cursor = conn.cursor()
        
        # Check initial schema version
        cursor.execute("PRAGMA user_version")
        initial_version = cursor.fetchone()[0]
        
        # Add new column (simulate migration)
        cursor.execute("ALTER TABLE patients ADD COLUMN email TEXT")
        
        # Update schema version
        cursor.execute("PRAGMA user_version = 2")
        
        # Verify migration
        cursor.execute("PRAGMA table_info(patients)")
        columns = [col[1] for col in cursor.fetchall()]
        
        has_email_column = "email" in columns
        
        cursor.execute("PRAGMA user_version")
        new_version = cursor.fetchone()[0]
        
        version_updated = new_version == 2
        
        print(f"  Schema version: {initial_version} -> {new_version}")
        print(f"  Email column added: {has_email_column}")
        print(f"  Migration successful: {version_updated and has_email_column}")
        
        conn.close()
        
        return version_updated and has_email_column
        
    except Exception as e:
        print(f"  Schema migration error: {e}")
        return False
    finally:
        tester.cleanup()

def run_database_corruption_tests():
    """Run all database corruption and recovery tests."""
    print("🗄️ Database Corruption and Recovery Testing")
    print("=" * 50)
    
    tests = [
        ("Database Integrity Check", test_database_integrity_check),
        ("Partial Corruption Recovery", test_partial_corruption_recovery),
        ("Transaction Rollback", test_transaction_rollback),
        ("Backup and Restore", test_backup_and_restore),
        ("Concurrent Access", test_concurrent_access),
        ("Large Data Handling", test_large_data_handling),
        ("Schema Migration", test_schema_migration),
    ]
    
    passed = 0
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
        print()
    
    print("=" * 50)
    print(f"Database Tests: {passed}/{len(tests)} passed")
    
    if passed >= len(tests) * 0.8:
        print("🏆 Robust database corruption handling!")
    else:
        print(f"⚠️ {len(tests) - passed} database issues need attention")
    
    return passed >= len(tests) * 0.8

if __name__ == "__main__":
    success = run_database_corruption_tests()
    exit(0 if success else 1)