#pragma once

#include <stdexcept>

__host__ __forceinline__ void check_cuda_error(cudaError_t err) {
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error " + std::to_string(err) + ": " + cudaGetErrorString(err));
    }
}

/*
launch_kernel() returns a functor (impl::kernel_launch_t) capturing the kernel launch configuration
impl::kernel_launch_t can be constexpr, invoking this functor launches the kernel
kernel_omit_bounds can be used if bounds are not needed by the kernel beyond bounds checking

launch_kernel<kernel, threads...>(bounds...)(args...) is equivalent to:
    kernel<<<
        dim3(((bounds + threads - 1) / threads)...),
        dim3(threads...)
    >>>(bounds..., idx..., args...);

launch_kernel<kernel, threads...>(kernel_omit_bounds, bounds...)(args...) is equivalent to:
    kernel<<<
        dim3(((bounds + threads - 1) / threads)...),
        dim3(threads...)
    >>>(idx..., args...);

where idx... = blockIdx.i * blockDim.i + threadIdx.i, with i = x/y/z

cudaDeviceSynchronize() is implicitly called after, and errors are checked for using check_cuda_error()

impl::kernel_wrapper() is the actual kernel launched, to wrap the call to kernel() and handle the following:
    idx... is calculated and checked against bounds...
    kernel() is only called if within bounds

kernel() needs to be declared using __device__ (not __global__) for this to work
*/

namespace impl {

// helpers for variadic expansion of blockIdx.i * blockDim.i + threadIdx.i

__device__ constexpr unsigned dim3::* dim[] = {
    &dim3::x,
    &dim3::y,
    &dim3::z
};

__device__ constexpr unsigned uint3::* idx[] = {
    &uint3::x,
    &uint3::y,
    &uint3::z
};

// helpers for passing bounds to kernel

template<std::size_t I>
struct bound_t
{
    int value;
};

template<typename IndexSequence>
struct kernel_bounds_impl_t;

template<std::size_t... Is>
struct kernel_bounds_impl_t<std::index_sequence<Is...>>
    : bound_t<Is>...
{
    template<typename... Args>
    __host__ __device__ __forceinline__ constexpr kernel_bounds_impl_t(Args... args) : bound_t<Is>{ args }... {}

    template<std::size_t I>
    __host__ __device__ __forceinline__ constexpr unsigned get() const {
        return bound_t<I>::value;
    }
};

template<std::size_t N>
using kernel_bounds_t = kernel_bounds_impl_t<std::make_index_sequence<N>>;

// wrapper kernel for idx... = blockIdx.i * blockDim.i + threadIdx.i boilerplate

template<auto Kernel, unsigned Dim, bool OmitBounds, std::size_t... Is, typename... Args>
__device__ __forceinline__ void kernel_wrapper_impl(std::index_sequence<Is...>, kernel_bounds_t<Dim> bounds, Args... args) {
    const unsigned i[] = { (blockIdx.*idx[Is] * blockDim.*dim[Is] + threadIdx.*idx[Is])... };
    if (((i[Is] < bounds.template get<Is>()) && ...)) {
        if constexpr (OmitBounds) {
            Kernel(i[Is]..., args...);
        } else {
            Kernel(bounds.template get<Is>()..., i[Is]..., args...);
        }
    }
}

template<auto Kernel, unsigned Dim, bool OmitBounds, typename... Args>
__global__ void kernel_wrapper(Args... args) {
    kernel_wrapper_impl<Kernel, Dim, OmitBounds>(std::make_index_sequence<Dim>{}, args...);
}

// functor for capturing bounds, can be constexpr

template<auto Kernel, bool OmitBounds, unsigned... Threads>
struct kernel_launch_t
{
    dim3 blocks;
    dim3 threads;
    kernel_bounds_t<sizeof...(Threads)> bounds;

    // CUDA does not support passing rvalue references to kernels
    template<typename... Args>
    __forceinline__ void operator()(Args... args) const {
        static_assert(
            OmitBounds
                ? std::is_invocable_v<decltype(Kernel), decltype(Threads)..., Args...>
                : std::is_invocable_v<decltype(Kernel), decltype(Threads)..., decltype(Threads)..., Args...>,
            "Kernel cannot be called with provided arguments (did you forget kernel_omit_bounds?)"
        );
        launch(std::make_index_sequence<sizeof...(Threads)>{}, args...);
    }

protected:
    template<std::size_t... Is, typename... Args>
    __forceinline__ void launch(std::index_sequence<Is...>, Args... args) const {
        impl::kernel_wrapper<Kernel, sizeof...(Threads), OmitBounds><<<blocks, threads>>>(bounds, args...);
        check_cuda_error(cudaDeviceSynchronize());
    }
};

// helper for launch_kernel

template<auto Kernel, bool OmitBounds, unsigned... Threads, std::size_t... Is, typename... Args>
__host__ __forceinline__ constexpr impl::kernel_launch_t<Kernel, OmitBounds, Threads...> launch_kernel_aux(std::index_sequence<Is...>, Args... args) {
    return { dim3(((args + Threads - 1) / Threads)...), dim3(Threads...), { std::forward<Args>(args)... } };
}

// kernel_omit_bounds tag struct
struct kernel_omit_bounds_t {};

}

// use this to not pass bounds to the kernel
constexpr impl::kernel_omit_bounds_t kernel_omit_bounds;

template<auto Kernel, unsigned... Threads, typename Arg0, typename... Args, bool OmitBounds = std::is_same_v<std::decay_t<Arg0>, impl::kernel_omit_bounds_t>>
__host__ __forceinline__ constexpr impl::kernel_launch_t<Kernel, OmitBounds, Threads...> launch_kernel(Arg0&& arg0, Args&&... args) {
    static_assert(sizeof...(Threads) <= 3, "More than 3 thread dimensions provided");

    // these two combined should ensure both are equal
    static_assert(sizeof...(Args) + (OmitBounds ? 0 : 1) >= sizeof...(Threads), "Bounds dimension less than thread dimension");
    static_assert(sizeof...(Args) + (OmitBounds ? 0 : 1) <= sizeof...(Threads), "Bounds dimension greater than thread dimension");

    if constexpr (OmitBounds) {
        return impl::launch_kernel_aux<Kernel, OmitBounds, Threads...>(std::make_index_sequence<sizeof...(Threads)>{}, std::forward<Args>(args)...);
    } else {
        return impl::launch_kernel_aux<Kernel, OmitBounds, Threads...>(std::make_index_sequence<sizeof...(Threads)>{}, std::forward<Arg0>(arg0), std::forward<Args>(args)...);
    }
}
