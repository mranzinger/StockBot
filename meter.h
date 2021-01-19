#pragma once

#include <torch/torch.h>

#include <unordered_map>
#include <vector>
#include <iostream>

template<typename T>
class Meter
{
public:
    Meter(bool resetOnValue = true) : m_resetOnValue(true) {}

    void Add(T value)
    {
        m_runningSum += value;
        ++m_count;
    }

    T Value(c10::optional<bool> reset = c10::optional<bool>()) const
    {
        if (DoReset(reset)) {
            throw std::runtime_error("Cannot call const accessor on this due to expected reset");
        }

        if (m_count == 0) return 0;

        return m_runningSum / m_count;
    }

    T Value(c10::optional<bool> reset = c10::optional<bool>())
    {
        T ret{0};
        if (m_count > 0) {
            ret = m_runningSum / m_count;
        }

        if (DoReset(reset)) {
            m_runningSum = 0;
            m_count = 0;
        }

        return ret;
    }

private:
    bool DoReset(c10::optional<bool> reset) const
    {
        return reset.value_or(m_resetOnValue);
    }

    bool m_resetOnValue;
    T m_runningSum{0};
    size_t m_count{0};
};

template<typename T>
class NamedMeter : public Meter<T>
{
public:
    NamedMeter(std::string name, bool resetOnValue)
        : Meter<T>(resetOnValue), m_name(std::move(name))
    {}

    const std::string &Name() const { return m_name; }

private:
    std::string m_name;
};

template<typename T>
class MeterDict;

template<typename T>
std::ostream &operator<<(std::ostream &os, MeterDict<T> &meters);

template<typename T>
class MeterDict
{
    template<typename K>
    friend std::ostream &operator<<(std::ostream &os, MeterDict<K> &meters);
public:
    typedef std::unordered_map<std::string, NamedMeter<T>> inner_map_t;

    MeterDict(bool resetOnValue = true) : m_resetOnValue(resetOnValue) {}

    void Add(const std::string &name, T value)
    {
        auto iter = m_meters.find(name);
        if (iter == m_meters.end()) {
            iter = m_meters.emplace(name, NamedMeter<T>(name, m_resetOnValue)).first;
            m_addOrder.push_back(name);
        }

        iter->second.Add(value);
    }

    T Value(const std::string &name, c10::optional<bool> reset = c10::optional<bool>())
    {
        auto iter = m_meters.find(name);
        if (iter == m_meters.end()) {
            throw std::runtime_error("The specified meter doesn't exist!");
        }

        return iter->second.Value(reset);
    }

    typename inner_map_t::iterator begin() { return m_meters.begin(); }
    typename inner_map_t::const_iterator begin() const { return m_meters.begin(); }

    typename inner_map_t::iterator end() { return m_meters.end(); }
    typename inner_map_t::const_iterator end() const { return m_meters.end(); }

    size_t size() const { return m_meters.size(); }
    bool empty() const { return m_meters.empty(); }

private:
    inner_map_t m_meters;
    bool m_resetOnValue;
    std::vector<std::string> m_addOrder;
};

template<typename T>
inline std::ostream &operator<<(std::ostream &os, NamedMeter<T> &meter)
{
    os << meter.Name() << ": " << meter.Value();
    return os;
}

template<typename T>
inline std::ostream &operator<<(std::ostream &os, MeterDict<T> &meters)
{
    os << "{";
    if (not meters.empty()) {
        bool first = true;
        for (auto &mName : meters.m_addOrder) {
            if (not first) {
                os << ", ";
            }

            os << meters.m_meters.find(mName)->second;

            first = false;
        }
    }

    os << "}";
    return os;
}