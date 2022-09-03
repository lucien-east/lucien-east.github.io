---
title: "[C++] Optimise the Efficiency when Using STL Containers - taking vector emplace_back as an example"
date: 2021-07-29 17:35:20
categories: [Programming Language, C++]
tags: [C++, Optimisation]
lang: en
mathjax:
---

C++ provides a plethora of containers that allow us to dynamically allocate our elements in run time. Take `vector` as an example. It is widely used in many applications. However, for some extreme cases, we may still want to avoid the overhead of reallocation or copy operations when those operations are quite expensive.

Since C++11, with the advent of `emplace` methods, developers are able to pursue more efficiency via them.

<!--more-->

## A Toy Example

To get a grasp of the usage of `emplace` , let's firstly look at a simple object. Say we have a class containing the information of a stock:

```C++ =
class Stock {
public:
    Stock(string date, double price, int volume) :
        date(date), price(price), volume(volume)
    {
        cout << "Contruct: " << date << "\n";
    }

    Stock(const Stock& obj) :
        date(obj.date), price(obj.price), volume(obj.volume)
    {
        cout << "Copied: " << date << "\n";
    }
private:
    string date;
    double price;
    int volume;
};
```

For the convenience of observation, when the object is created or copied, the corresponded message will be printed.

## The Behavior of Copy between `push_back` and `emplace_back`

We now use a vector - `portfolio` -  to maintain our stock objects.

First, we add the object `2021-08-01` into it via `push_back()`:

```C++ =
int main()
{
    vector<Stock> portfolio;
    portfolio.push_back({"2021-08-01", 10.0, 3});

    return 0;
}
```

When executing the program, we get the output:

```
Contruct: 2021-08-01
Copied: 2021-08-01
```

It shows that this object is created and copied into the vector afterwards. The copy occurs for the reason that our object is not allocated on the space where our vector `portfolio` is located. The system still has to copy it onto the vector space.

This copy operation can be spared if our system can simply construct our object onto `portfolio`'s space. The method `emplace_back()` can come into play:

```C++ =
vector<Stock> portfolio;
portfolio.emplace_back("2021-08-01", 10.0, 3);
```

When executing the program, the copy operation has disappeared. The object now is constructed in place:

```
Contruct: 2021-08-01
```

Note the syntax sugar only needs us to specify our arguments: [^1]
```C++
template <class... Args>
    void emplace_back (Args&&... args);
```


## Another Unexpected Copy behind the Background

Cautious readers would find that if we add more items, there might be still copy occurring:

```C++ =
vector<Stock> portfolio;
portfolio.emplace_back("2021-08-01", 10.0, 3);
portfolio.emplace_back("2021-08-02", 12.5, 5);
portfolio.emplace_back("2021-08-03", 15.7, 1);
```

Its output:

```
Contruct: 2021-08-01
Contruct: 2021-08-02
Copied: 2021-08-01
Contruct: 2021-08-03
Copied: 2021-08-01
Copied: 2021-08-02
```

To understand this, we have to know that the **storage space** of a vector ought to be allocated in advance. Do not confuse this storage space with the **actual size** used by the user (indicated by `size()`). The storage space is always equal or greater than the atual size of a vector, so that our system need not reallocate on each insertion. The size of the storage space can be obtained by the method `capacity()`.

Now we are able to analyse this behavior with the help of it:

```C++ =
vector<Stock> portfolio;

cout << "Capacity before inserting 2021-08-01 = " << portfolio.capacity() << "\n";
portfolio.emplace_back("2021-08-01", 10.0, 3);

cout << "Capacity before inserting 2021-08-02 = " << portfolio.capacity() << "\n";
portfolio.emplace_back("2021-08-02", 12.5, 5);

cout << "Capacity before inserting 2021-08-03 = " << portfolio.capacity() << "\n";
portfolio.emplace_back("2021-08-03", 15.7, 1);
```

Output:

```
Capacity before inserting 2021-08-01 = 0
Contruct: 2021-08-01

Capacity before inserting 2021-08-02 = 1
Contruct: 2021-08-02
Copied: 2021-08-01

Capacity before inserting 2021-08-03 = 2
Contruct: 2021-08-03
Copied: 2021-08-01
Copied: 2021-08-02
```

When `2021-08-01` is constructed, it just allocates in place, no copy is needed as aforementioned.

However, when adding `2021-08-02`, the acutual size of a vector now grows to 2, the storage space is not enough to accomodate. Thus, the system has to reallocate its storage space in advance, then **constructing** `2021-08-02` on it, followed by **copying** the `2021-08-01` objects into the newly allocated space.

Similarly, `2021-08-03` is **constructed in place** before the system allocates enough storage space. Then,`2021-08-01` and `2021-08-02` will be **copied** into the reallocated space afterwards.

> Note the elements in a vector are stored in a **contiguous memory block**.[^2] This is the reason that when there is no enough storage space for insertion, the system has to reallocate a memory block to arrange its elements. The behavior of how capacity will grow may vary by different system.


## The Usage of `reserve`

To optimise this, we could have our vector contain enough storage space in advance. This is the moment when the method `reserve()` can take advantage:

```C++ =
vector<Stock> portfolio;

portfolio.reserve(5);   // Request enough storage space

cout << "Capacity before inserting 2021-08-01 = " << portfolio.capacity() << "\n";
portfolio.emplace_back("2021-08-01", 10.0, 3);

cout << "Capacity before inserting 2021-08-02 = " << portfolio.capacity() << "\n";
portfolio.emplace_back("2021-08-02", 12.5, 5);

cout << "Capacity before inserting 2021-08-03 = " << portfolio.capacity() << "\n";
portfolio.emplace_back("2021-08-03", 15.7, 1);
```

Output:

```
Capacity before inserting 2021-08-01 = 5
Contruct: 2021-08-01

Capacity before inserting 2021-08-02 = 5
Contruct: 2021-08-02

Capacity before inserting 2021-08-03 = 5
Contruct: 2021-08-03
```

The copy operations has disappeared if the storage size are large enough. Remember that `reserve()` does not cause any effect on the vector size. 

> The maximum capacity of a vector is restricted by `vector::max_size`.


[^1]: [C++ Reference - std::vector::emplace_back](https://www.cplusplus.com/reference/vector/vector/emplace_back/)

[^2]: [C++ Standard n4140: Class template vector overview](https://timsong-cpp.github.io/cppwp/n4140/vector.overview#1)