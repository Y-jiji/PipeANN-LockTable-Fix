#include <atomic>
#include <cstdlib>
#include <chrono>
#include <future>
#include <iostream>
#include <thread>
#include <vector>
#include "utils/lock_table.h"

using Table = pipeann::SparseLockTable<unsigned>;
using namespace std::chrono_literals;

void check(bool condition) {
  if (!condition) std::abort();
}

void test_spin_lock() {
  pipeann::RWSpinLock lock;
  check(lock.tryrdlock() == 0);
  auto reader = std::async(std::launch::async, [&] {
    check(lock.tryrdlock() == 0);
    check(lock.trywrlock() == EBUSY);
    lock.unlock();
  });
  check(reader.wait_for(5s) == std::future_status::ready);
  reader.get();
  lock.unlock();
  check(lock.trywrlock() == 0);
  check(lock.trywrlock() == EBUSY);
  check(lock.tryrdlock() == EBUSY);
  std::thread releaser([&] { lock.unlock(); });
  releaser.join();

  // Test the atomic latch directly, without the map's synchronization masking races.
  unsigned value = 0;
  std::vector<std::future<void>> workers;
  for (unsigned t = 0; t < 8; ++t) {
    workers.push_back(std::async(std::launch::async, [&] {
      for (unsigned i = 0; i < 10000; ++i) {
        lock.wrlock();
        ++value;
        lock.unlock();
        lock.rdlock();
        unsigned before = value;
        std::this_thread::yield();
        check(value == before);
        lock.unlock();
      }
    }));
  }
  for (auto &worker : workers) {
    check(worker.wait_for(30s) == std::future_status::ready);
    worker.get();
  }
  check(value == 80000);
}

void test_cross_thread_release() {
  // A background thread releases a lock acquired elsewhere, with writers already waiting.
  // Exercise both read and write holds; the table must retain queued users' entries.
  for (bool read : {false, true}) {
    Table table;
    if (read) table.rdlock(42); else table.wrlock(42);
    std::atomic<unsigned> started{0};
    std::vector<std::future<void>> waiters;
    for (unsigned i = 0; i < 8; ++i) {
      waiters.push_back(std::async(std::launch::async, [&] {
        ++started;
        table.wrlock(42);
        table.unlock(42);
      }));
    }
    while (started != 8) std::this_thread::yield();
    std::this_thread::sleep_for(20ms);
    for (auto &waiter : waiters) check(waiter.wait_for(0s) == std::future_status::timeout);
    std::thread background([&] { table.unlock(42); });
    background.join();
    for (auto &waiter : waiters) {
      check(waiter.wait_for(10s) == std::future_status::ready);
      waiter.get();
    }
    check(table.size() == 0);
  }
}

int main() {
  // Keep deadlock regressions bounded even if unlock itself gets stuck.
  std::atomic<bool> finished{false};
  std::thread watchdog([&] {
    for (unsigned i = 0; i < 600 && !finished; ++i) std::this_thread::sleep_for(100ms);
    check(finished);
  });
  test_spin_lock();
  test_cross_thread_release();
  Table table;
  // Concurrent readers must be admitted; writers must be excluded.
  table.rdlock(7);
  auto reader = std::async(std::launch::async, [&] {
    check(table.tryrdlock(7) == 0);
    check(table.trywrlock(7) != 0);
    table.unlock(7);
  });
  check(reader.wait_for(5s) == std::future_status::ready);
  reader.get();
  table.unlock(7);
  check(table.size() == 0);

  // A waiter must not prevent its owner from reaching unlock; queued users keep the entry alive.
  for (bool read : {false, true}) {
    table.wrlock(9);
    std::atomic<int> started{0};
    std::vector<std::future<void>> waiters;
    for (int i = 0; i < 16; ++i) {
      waiters.push_back(std::async(std::launch::async, [&] {
        ++started;
        if (read) table.rdlock(9); else table.wrlock(9);
        table.unlock(9);
      }));
    }
    while (started != 16) std::this_thread::yield();
    std::this_thread::sleep_for(20ms);
    for (auto &w : waiters) check(w.wait_for(0s) == std::future_status::timeout);
    check(table.tryrdlock(9) != 0);
    check(table.trywrlock(9) != 0);
    table.unlock(9);
    for (auto &w : waiters) {
      check(w.wait_for(10s) == std::future_status::ready);
      w.get();
    }
    check(table.size() == 0);
  }

  // Mixed blocking/try users repeatedly retire and recreate entries under contention.
  unsigned values[4]{};
  std::vector<std::future<void>> workers;
  for (unsigned t = 0; t < 16; ++t) {
    workers.push_back(std::async(std::launch::async, [&, t] {
      for (unsigned i = 0; i < 10000; ++i) {
        unsigned key = i % 4;
        if (t % 2) {
          while (table.trywrlock(key) != 0) std::this_thread::yield();
        } else {
          table.wrlock(key);
        }
        ++values[key];
        table.unlock(key);
        table.rdlock(key);
        unsigned before = values[key];
        std::this_thread::yield();
        check(values[key] == before);
        table.unlock(key);
      }
    }));
  }
  for (auto &w : workers) {
    check(w.wait_for(30s) == std::future_status::ready);
    w.get();
  }
  for (unsigned value : values) check(value == 40000);
  check(table.size() == 0);
  finished = true;
  watchdog.join();
  std::cout << "PASS: cross-thread release, atomic latch, shared reads, exclusion, queued readers/writers, mixed try/blocking contention, entry reclamation\n";
}
